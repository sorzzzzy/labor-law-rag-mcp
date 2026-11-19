import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.ConnectException
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.net.http.HttpTimeoutException
import java.nio.charset.StandardCharsets
import java.time.Duration

/**
 * 어제 만든 Python 검색 API(GET /search)를 호출하는 클라이언트.
 *
 * MCP 서버는 검색을 직접 하지 않고, 검색 API에 맡긴 뒤 결과를 Claude가 읽기 좋은 텍스트로 바꿔서 돌려준다.
 * (검색 로직은 Python 쪽 한 곳에만 두고, MCP 서버는 "AI 앱과 검색 API를 잇는 어댑터" 역할만 한다)
 */
class LaborLawSearchClient(private val baseUrl: String) {

    // JDK 기본 HTTP 클라이언트. 별도 라이브러리 없이 쓸 수 있다 (Spring의 RestClient 같은 역할)
    private val http: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(3))
        .build()

    /**
     * 질문으로 검색해서, Claude에게 돌려줄 텍스트를 만든다.
     * 실패하면 예외 대신 Result.failure로 돌려줘서, 도구 쪽에서 에러 메시지로 바꿔 보여줄 수 있게 한다.
     */
    suspend fun search(query: String, k: Int): Result<String> = runCatching {
        // 한글과 띄어쓰기를 URL에 넣을 수 있게 인코딩 (curl의 --data-urlencode와 같은 일)
        val q = URLEncoder.encode(query, StandardCharsets.UTF_8)
        val request = HttpRequest.newBuilder(URI.create("$baseUrl?q=$q&k=$k"))
            .timeout(Duration.ofSeconds(30))
            .GET()
            .build()

        // HTTP 호출은 기다리는(블로킹) 작업이라 IO 전용 스레드에서 실행한다
        val response = withContext(Dispatchers.IO) {
            http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        }
        // 200이 아니면 상태 코드를 담은 예외를 던진다 (failureMessage에서 4xx/5xx를 나눠 안내하려고)
        if (response.statusCode() != 200) throw SearchApiStatusException(response.statusCode())

        format(query, Json.parseToJsonElement(response.body()).jsonObject)
    }

    /**
     * 실패 원인을 Claude가 읽을 에러 문구로 바꾼다.
     *
     * 에러 문구도 Claude가 읽고 다음 행동을 정하는 입력이다. 그래서 세 가지를 담는다.
     *   1) 원인: 연결 실패인지, 시간 초과인지, 요청 값 문제인지, 서버 내부 오류인지
     *   2) 다시 시도할 만한지
     *   3) 끝내 실패하면 어떻게 할지: 다른 출처로 추측하지 말고 사용자에게 먼저 알릴 것
     * 이 문구는 사용자에게 그대로 전달될 수 있어서, 서버 이름(uvicorn)이나 예외 이름 같은 운영자용 정보는 넣지 않는다.
     * (운영자용 정보는 Main.kt에서 로그로 따로 남긴다)
     */
    fun failureMessage(e: Throwable): String {
        val (cause, retry) = when (e) {
            // 검색 API가 꺼져 있음.
            // JDK HttpClient의 ConnectException은 message가 null인 경우가 있어서 e.message를 그대로 쓰지 않는다
            // (수정 전에는 "호출에 실패했습니다: null"로 찍혔다)
            is ConnectException -> "근로기준법 검색 시스템에 연결할 수 없습니다." to NO_RETRY
            // 연결(3초) 또는 응답(30초) 시간 초과. 일시적인 지연일 수 있다
            is HttpTimeoutException -> "근로기준법 검색 시스템의 응답 시간이 초과되었습니다." to RETRY_ONCE
            // 4xx는 요청 값 문제(고쳐서 다시 보내면 됨), 5xx는 검색 시스템 내부 문제(예: Elasticsearch가 꺼짐)
            is SearchApiStatusException ->
                if (e.status in 400..499) {
                    "검색 요청 값이 올바르지 않습니다(응답 코드 ${e.status}). query는 비어 있지 않은 문자열, k는 1~10 사이 정수여야 합니다." to RETRY_ONCE
                } else {
                    "근로기준법 검색 시스템 내부에서 오류가 발생했습니다(응답 코드 ${e.status})." to NO_RETRY
                }
            else -> "근로기준법 검색 중 알 수 없는 오류가 발생했습니다." to NO_RETRY
        }
        return "검색 실패: $cause $retry $ON_FAILURE"
    }

    /**
     * 검색 API의 JSON 응답을 사람이(그리고 Claude가) 읽기 좋은 텍스트로 바꾼다.
     *
     * 예)
     * [1] 근로기준법 제26조 (해고의 예고)
     * 출처: 근로기준법 제26조 (시행 2026. 10. 8., 법률 제21533호)
     * 원문:
     * 사용자는 근로자를 해고하려면 적어도 30일 전에 예고를 하여야 하고, ...
     */
    private fun format(query: String, body: JsonObject): String {
        val results = body["results"]?.jsonArray.orEmpty()
        if (results.isEmpty()) return "'$query'에 대한 근로기준법 검색 결과가 없습니다."

        val items = results.mapIndexed { index, element ->
            val r = element.jsonObject
            fun field(name: String) = r[name]?.jsonPrimitive?.contentOrNull

            val article = field("article") ?: ""
            val paragraph = field("paragraph")?.let { " ${it}항" } ?: ""
            val title = field("title")?.let { " ($it)" } ?: ""
            """
            |[${index + 1}] 근로기준법 $article$paragraph$title
            |출처: ${field("source")}
            |원문:
            |${field("text")}
            """.trimMargin()
        }

        return buildString {
            appendLine("검색어: $query")
            appendLine("근로기준법 검색 결과 ${results.size}건")
            appendLine()
            append(items.joinToString("\n\n"))
        }
    }
}

/** 검색 API가 200이 아닌 응답을 돌려줬을 때 던지는 예외. 상태 코드를 보고 에러 문구를 고르려고 따로 둔다 */
class SearchApiStatusException(val status: Int) : RuntimeException("검색 API 응답 코드 $status")

// 에러 문구에 붙일 안내 (failureMessage에서 사용)
private const val NO_RETRY = "다시 시도해도 같은 결과일 가능성이 높으니 재시도하지 마세요."
private const val RETRY_ONCE = "일시적인 문제일 수 있으니 한 번만 다시 시도하세요."
private const val ON_FAILURE =
    "검색에 실패한 상태에서는 웹 검색이나 기억으로 조문을 대신 찾지 말고, " +
        "지금은 근로기준법 근거 조항을 확인할 수 없다는 사실을 답변 첫머리에서 사용자에게 먼저 알리세요."
