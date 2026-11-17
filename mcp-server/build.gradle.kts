plugins {
    // MCP Kotlin SDK 0.16.0이 Kotlin 2.4로 빌드되어 있어 같은 버전을 쓴다
    kotlin("jvm") version "2.4.21"
    // installDist로 실행 스크립트(bin/labor-law-mcp)를 만들기 위해
    application
}

group = "dev.jinsol"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    // MCP 서버 API: Server, StdioServerTransport, addTool 등
    implementation("io.modelcontextprotocol:kotlin-sdk-server:0.16.0")

    // 코루틴, stdio 입출력, JSON 파싱 (SDK가 쓰는 버전에 맞춤)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.9.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.12.0")

    // 로그는 stderr로 보낸다. stdout은 MCP 메시지(JSON-RPC) 전용이라 다른 출력이 섞이면 통신이 깨진다
    implementation("org.slf4j:slf4j-simple:2.0.20")
    runtimeOnly("io.github.oshai:kotlin-logging-jvm:8.0.4")
}

application {
    // src/main/kotlin/Main.kt 의 main 함수
    mainClass.set("MainKt")
}
