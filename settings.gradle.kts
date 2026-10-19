pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "ug-cuda-kotlin"

listOf(
    "../half-kotlin",
    "../cudarc-kotlin",
    "../ug-kotlin",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}
