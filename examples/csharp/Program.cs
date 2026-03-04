// Aura FFI example for C# (.NET 8+).
//
// Demonstrates calling the Aura shared library from C# via P/Invoke.
//
// Build the shared library first:
//   cargo build --release --no-default-features --features "encryption,ffi"
//
// Then run:
//   dotnet run
//
// Make sure aura.dll (Windows) / libaura.so (Linux) / libaura.dylib (macOS)
// is in the output directory or system library path.

using System;
using System.Runtime.InteropServices;

// ── P/Invoke declarations ──

static class AuraFFI
{
    const string LibName = "aura";

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aura_open(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        out IntPtr outError);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aura_close(IntPtr handle, out IntPtr outError);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aura_free(IntPtr handle);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aura_free_string(IntPtr s);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aura_store(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string content,
        byte level,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? tagsJson,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? ns,
        out IntPtr outError);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aura_recall(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
        int tokenBudget,
        out IntPtr outError);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aura_recall_structured(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
        int topK,
        out IntPtr outError);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aura_run_maintenance(IntPtr handle, out IntPtr outError);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern long aura_count(IntPtr handle);

    // Helper: read and free a Rust string
    public static string? ReadAndFree(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return null;
        string result = Marshal.PtrToStringUTF8(ptr)!;
        aura_free_string(ptr);
        return result;
    }

    public static string? GetError(IntPtr errPtr)
    {
        return ReadAndFree(errPtr);
    }
}

// ── Main ──

class Program
{
    static void Main()
    {
        Console.WriteLine("=== Aura C# FFI Example ===");

        // Open brain
        var handle = AuraFFI.aura_open("./csharp_brain", out var err);
        if (handle == IntPtr.Zero)
        {
            Console.WriteLine($"Error: {AuraFFI.GetError(err)}");
            return;
        }

        try
        {
            // Store memories
            var id1 = AuraFFI.ReadAndFree(
                AuraFFI.aura_store(handle, "User prefers C# and .NET",
                    4, // Identity
                    "[\"preference\",\"dotnet\"]", null, out err));
            Console.WriteLine($"Stored: {id1}");

            var id2 = AuraFFI.ReadAndFree(
                AuraFFI.aura_store(handle, "Always write unit tests first",
                    2, // Decisions
                    "[\"workflow\",\"testing\"]", null, out err));
            Console.WriteLine($"Stored: {id2}");

            // Count
            var count = AuraFFI.aura_count(handle);
            Console.WriteLine($"Total records: {count}");

            // Recall (text)
            var context = AuraFFI.ReadAndFree(
                AuraFFI.aura_recall(handle, "user preferences", 1024, out err));
            Console.WriteLine($"\nRecall (text):\n{context}");

            // Recall structured (JSON)
            var json = AuraFFI.ReadAndFree(
                AuraFFI.aura_recall_structured(handle, "testing workflow", 5, out err));
            Console.WriteLine($"\nRecall structured (JSON):\n{json}");

            // Maintenance
            AuraFFI.aura_run_maintenance(handle, out err);
            Console.WriteLine("\nMaintenance cycle complete.");
        }
        finally
        {
            AuraFFI.aura_close(handle, out err);
            AuraFFI.aura_free(handle);
        }
    }
}
