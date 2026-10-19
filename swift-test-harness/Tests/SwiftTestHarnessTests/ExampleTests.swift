#if canImport(Testing)
import Testing
import UgCuda

@Suite("UgCuda Swift Export Tests")
struct UgCudaExportTests {
    @Test("UgCuda swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "UgCuda swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import UgCuda

final class UgCudaExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "UgCuda swift module imported cleanly")
    }
}
#endif
