// Aura FFI example for Go.
//
// Demonstrates calling the Aura shared library from Go via CGo.
//
// Build the shared library first:
//   cargo build --release --no-default-features --features "encryption,ffi"
//
// Then run:
//   # Linux/macOS
//   CGO_LDFLAGS="-L../../target/release -laura" go run main.go
//
//   # Windows
//   CGO_LDFLAGS="-L../../target/release -laura" go run main.go
//
// Make sure the shared library (.so/.dylib/.dll) is in your library path.

package main

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -laura
#include <stdlib.h>

// ── Aura C FFI declarations ──

typedef void* AuraHandle;

AuraHandle aura_open(const char* path, char** out_error);
int aura_close(AuraHandle handle, char** out_error);
void aura_free(AuraHandle handle);
void aura_free_string(char* s);

// level: 1=Working, 2=Decisions, 3=Domain, 4=Identity, 0=default
char* aura_store(AuraHandle handle, const char* content, unsigned char level,
                 const char* tags_json, const char* namespace, char** out_error);

char* aura_recall(AuraHandle handle, const char* query,
                  int token_budget, char** out_error);

char* aura_recall_structured(AuraHandle handle, const char* query,
                             int top_k, char** out_error);

int aura_run_maintenance(AuraHandle handle, char** out_error);
long long aura_count(AuraHandle handle);
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

func main() {
	brainPath := C.CString("./go_brain")
	defer C.free(unsafe.Pointer(brainPath))

	var errPtr *C.char

	// ── Open ──
	handle := C.aura_open(brainPath, &errPtr)
	if handle == nil {
		fmt.Printf("Error opening brain: %s\n", C.GoString(errPtr))
		C.aura_free_string(errPtr)
		os.Exit(1)
	}
	defer func() {
		C.aura_close(handle, &errPtr)
		C.aura_free(handle)
	}()

	fmt.Println("=== Aura Go FFI Example ===")

	// ── Store ──
	content := C.CString("User prefers dark mode and uses Vim")
	level := C.uchar(4) // Identity
	tags := C.CString(`["preference","editor"]`)
	var ns *C.char // NULL = default namespace

	rid := C.aura_store(handle, content, level, tags, ns, &errPtr)
	if rid == nil {
		fmt.Printf("Store error: %s\n", C.GoString(errPtr))
		C.aura_free_string(errPtr)
	} else {
		fmt.Printf("Stored record: %s\n", C.GoString(rid))
		C.aura_free_string(rid)
	}
	C.free(unsafe.Pointer(content))
	C.free(unsafe.Pointer(tags))

	// Store more
	content2 := C.CString("Deploy to staging before production")
	tags2 := C.CString(`["workflow"]`)
	rid2 := C.aura_store(handle, content2, C.uchar(2), tags2, ns, &errPtr)
	if rid2 != nil {
		fmt.Printf("Stored record: %s\n", C.GoString(rid2))
		C.aura_free_string(rid2)
	}
	C.free(unsafe.Pointer(content2))
	C.free(unsafe.Pointer(tags2))

	// ── Count ──
	count := C.aura_count(handle)
	fmt.Printf("Total records: %d\n", count)

	// ── Recall (text) ──
	query := C.CString("user preferences")
	result := C.aura_recall(handle, query, 1024, &errPtr)
	if result != nil {
		fmt.Printf("\nRecall (text):\n%s\n", C.GoString(result))
		C.aura_free_string(result)
	}
	C.free(unsafe.Pointer(query))

	// ── Recall structured (JSON) ──
	query2 := C.CString("deployment workflow")
	jsonResult := C.aura_recall_structured(handle, query2, 5, &errPtr)
	if jsonResult != nil {
		fmt.Printf("\nRecall structured (JSON):\n%s\n", C.GoString(jsonResult))
		C.aura_free_string(jsonResult)
	}
	C.free(unsafe.Pointer(query2))

	// ── Maintenance ──
	C.aura_run_maintenance(handle, &errPtr)
	fmt.Println("\nMaintenance cycle complete.")
}
