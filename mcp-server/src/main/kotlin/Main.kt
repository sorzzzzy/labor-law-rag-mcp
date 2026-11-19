import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.server.ServerOptions
import io.modelcontextprotocol.kotlin.sdk.server.StdioServerTransport
import io.modelcontextprotocol.kotlin.sdk.types.CallToolResult
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.ServerCapabilities
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import io.modelcontextprotocol.kotlin.sdk.types.ToolAnnotations
import io.modelcontextprotocol.kotlin.sdk.types.ToolSchema
import kotlinx.coroutines.Job
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSink
import kotlinx.io.asSource
import kotlinx.io.buffered
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import org.slf4j.LoggerFactory

/** 어제 만든 검색 API 주소 (uvicorn api:app --port 8000) */
const val SEARCH_API_URL = "http://localhost:8000/search"

/** 운영자용 로그. slf4j-simple이 stderr로 출력하고, Claude 앱이 이를 MCP 로그 파일에 남긴다 */
private val log = LoggerFactory.getLogger("labor-law")

/**
 * 도구 설명 문구.
 * LLM은 도구의 이름과 설명만 보고 "언제 이 도구를 쓸지", "검색어를 어떻게 넣을지"를 판단한다.
 * 설명을 바꿔가며 Claude의 행동이 어떻게 달라지는지 비교하려고 두 버전을 둔다.
 */
data class ToolText(val description: String, val queryDescription: String)

/** 1차: 최소한의 설명 */
val TOOL_TEXT_V1 = ToolText(
    description = "근로기준법 조항을 검색합니다.",
    queryDescription = "검색어",
)

/**
 * 2차: 언제 쓰는지, 검색어를 어떻게 만드는지, 답변에 무엇을 붙이는지까지 알려주는 설명
 * (V1 실험에서 Claude가 검색어 하나에 주제 4개를 넣어 정답 조항이 6위로 밀려난 것을 보고,
 *  "주제마다 나눠서 검색" 문장을 추가했다)
 */
val TOOL_TEXT_V2 = ToolText(
    description = """
        대한민국 근로기준법(시행 2026. 10. 8., 법률 제21533호)의 조문을 검색해 관련 조항의 원문과 출처를 돌려줍니다.
        근로시간, 휴가·연차, 임금·수당, 해고, 직장 내 괴롭힘처럼 근로기준법과 관련된 질문에는
        기억에 의존하지 말고 반드시 이 도구로 근거 조항을 찾은 뒤 답하세요.
        사용자의 일상 표현은 법률 용어로 바꿔 검색하면 더 정확합니다.
        (예: '잘렸다' → '해고 예고', '야근 수당' → '연장근로 가산임금', '따돌림' → '직장 내 괴롭힘')
        질문에 주제가 여러 개 섞여 있으면 주제마다 나눠서 따로 검색하세요. 한 검색어에 여러 주제를 넣으면 관련 조항이 결과에서 밀려날 수 있습니다.
        답변에는 근거로 쓴 조항의 출처(조·항)를 함께 표시하세요.
    """.trimIndent(),
    queryDescription = "검색할 내용. 한 번에 한 주제만, 법률 용어가 들어간 짧은 문장이나 키워드로 (예: '해고 예고 기간')",
)

/** 지금 쓸 설명. 실험할 때 V1 ↔ V2를 바꾸고 다시 빌드한다 */
val TOOL_TEXT = TOOL_TEXT_V2

fun main() {
    // 1) MCP 서버 정보와 지원 기능 선언 (이 서버는 도구(tools)만 제공한다)
    val server = Server(
        serverInfo = Implementation(name = "labor-law", version = "0.1.0"),
        options = ServerOptions(
            capabilities = ServerCapabilities(tools = ServerCapabilities.Tools(listChanged = false)),
        ),
    )

    val searchClient = LaborLawSearchClient(SEARCH_API_URL)

    // 2) 도구 등록: 이름, 설명, 입력 스키마(JSON Schema), 그리고 호출됐을 때 실행할 코드
    //    Claude가 tools/list로 받아가는 정보가 바로 name + description + inputSchema 다
    server.addTool(
        name = "search_labor_law",
        description = TOOL_TEXT.description,
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                putJsonObject("query") {
                    put("type", "string")
                    put("description", TOOL_TEXT.queryDescription)
                }
                putJsonObject("k") {
                    put("type", "integer")
                    put("description", "돌려받을 조항 수 (기본 5)")
                    put("minimum", 1)
                    put("maximum", 10)
                }
            },
            required = listOf("query"),
        ),
        // 읽기 전용 도구라는 힌트: 데이터를 바꾸지 않으니 호스트 앱이 안심하고 부를 수 있다
        toolAnnotations = ToolAnnotations(readOnlyHint = true, openWorldHint = false),
    ) { request ->
        // Claude가 보낸 인자 꺼내기 (tools/call의 arguments)
        val query = request.arguments?.get("query")?.jsonPrimitive?.contentOrNull
        if (query.isNullOrBlank()) {
            return@addTool CallToolResult(
                content = listOf(TextContent("검색 실패: query 값이 비어 있습니다. 검색할 내용을 넣어 다시 요청하세요.")),
                isError = true,
            )
        }
        val k = request.arguments?.get("k")?.jsonPrimitive?.intOrNull ?: 5

        // 검색 API 호출 → 성공하면 결과 텍스트, 실패하면 원인을 알려주는 에러 결과
        searchClient.search(query, k).fold(
            onSuccess = { text -> CallToolResult(content = listOf(TextContent(text))) },
            onFailure = { e ->
                // 운영자용 상세 원인(예외 이름, 메시지, 확인할 곳)은 로그로만 남긴다
                log.warn(
                    "검색 API 호출 실패: query='{}', {}: {} (검색 API(uvicorn)가 실행 중인지 확인)",
                    query, e::class.java.name, e.message,
                )
                // Claude에게는 원인, 다시 시도할 만한지, 실패했을 때 할 일만 담은 문구를 돌려준다
                CallToolResult(
                    content = listOf(TextContent(searchClient.failureMessage(e))),
                    isError = true,
                )
            },
        )
    }

    // 3) stdio 연결: Claude 데스크톱 앱이 이 프로그램을 실행하고, 표준 입력/출력으로 JSON-RPC 메시지를 주고받는다
    //    그래서 println()으로 stdout에 아무거나 찍으면 메시지가 깨진다 (로그는 stderr로)
    val transport = StdioServerTransport(
        input = System.`in`.asSource().buffered(),
        output = System.out.asSink().buffered(),
    )

    // 4) 연결이 끊길 때까지 프로그램을 살려둔다 (Claude 앱이 종료하면 onClose가 불리고 끝난다)
    runBlocking {
        val session = server.createSession(transport)
        val done = Job()
        session.onClose { done.complete() }
        done.join()
    }
}
