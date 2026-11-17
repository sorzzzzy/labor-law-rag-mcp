import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
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
        check(response.statusCode() == 200) { "검색 API 응답 코드 ${response.statusCode()}" }

        format(query, Json.parseToJsonElement(response.body()).jsonObject)
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
