//! Aura Dashboard — standalone HTTP server with Memory Management UI.
//!
//! Download from GitHub Releases and run:
//!
//!     aura-dashboard ./my_brain
//!     aura-dashboard ./my_brain --port 9000
//!     aura-dashboard --help
//!
//! Environment variables (all optional):
//!     AURA_API_KEY        — Bearer token for API auth
//!     AURA_RATE_LIMIT     — Max requests/sec (0 = disabled)
//!     AURA_BIND           — Bind address override (e.g. 0.0.0.0:8080)
//!     AURA_CORS_ORIGINS   — Comma-separated origins (* = all)
//!     AURA_TLS_CERT       — Path to TLS certificate (enables HTTPS)
//!     AURA_TLS_KEY        — Path to TLS private key
//!     AURA_LOG_JSON       — Set to "1" for JSON log output
//!     RUST_LOG            — Tracing filter (info, debug, etc.)

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut path = String::from("./aura_brain");
    let mut port: u16 = 8000;
    let mut show_help = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                show_help = true;
                break;
            }
            "-p" | "--port" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Error: invalid port number '{}'", args[i]);
                        std::process::exit(1);
                    });
                } else {
                    eprintln!("Error: --port requires a value");
                    std::process::exit(1);
                }
            }
            "-V" | "--version" => {
                println!("aura-dashboard {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            arg if arg.starts_with('-') => {
                eprintln!("Unknown option: {}", arg);
                eprintln!("Run with --help for usage info.");
                std::process::exit(1);
            }
            _ => {
                path = args[i].clone();
            }
        }
        i += 1;
    }

    if show_help {
        println!("Aura Dashboard v{}", env!("CARGO_PKG_VERSION"));
        println!("Cognitive memory management UI for AuraSDK.\n");
        println!("USAGE:");
        println!("    aura-dashboard [PATH] [OPTIONS]\n");
        println!("ARGS:");
        println!("    PATH    Path to brain data directory (default: ./aura_brain)\n");
        println!("OPTIONS:");
        println!("    -p, --port <PORT>    Server port (default: 8000)");
        println!("    -V, --version        Print version");
        println!("    -h, --help           Print this help\n");
        println!("ENVIRONMENT:");
        println!("    AURA_API_KEY         Bearer token for API authentication");
        println!("    AURA_RATE_LIMIT      Max requests per second (0 = disabled)");
        println!("    AURA_BIND            Bind address (overrides --port)");
        println!("    AURA_CORS_ORIGINS    CORS origins (* = allow all)");
        println!("    AURA_TLS_CERT        TLS certificate path (enables HTTPS)");
        println!("    AURA_TLS_KEY         TLS private key path");
        println!("    AURA_LOG_JSON        Set to 1 for JSON logs");
        println!("    RUST_LOG             Log level filter (info, debug, trace)\n");
        println!("EXAMPLES:");
        println!("    aura-dashboard ./my_brain");
        println!("    aura-dashboard ./my_brain --port 9000");
        println!("    AURA_API_KEY=secret aura-dashboard ./data");
        return;
    }

    if let Err(e) = aura::server::start_server(port, &path) {
        eprintln!("Fatal: {}", e);
        std::process::exit(1);
    }
}
