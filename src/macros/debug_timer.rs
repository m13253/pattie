//! Timing utilities.

#[macro_export]
macro_rules! start_debug_timer {
    ($enabled:expr) => {{
        use ::core::option::Option::{None, Some};
        use ::core::primitive::bool;
        use ::std::time::Instant;
        let enabled: bool = $enabled;
        if enabled {
            Some(Instant::now())
        } else {
            None
        }
    }};
}

#[macro_export]
macro_rules! print_debug_timer {
    ($timer:expr, $($step_name_fmt:expr),+) => {{
        use ::core::option::Option::{self, Some};
        use ::std::time::Instant;
        let timer: Option<Instant> = $timer;
        if let Some(start) = timer {
            let duration = Instant::now().duration_since(start);
            ::std::eprintln!(
                "[Timing] {}:\t{}.{:09} seconds",
                ::core::format_args!($($step_name_fmt),+),
                duration.as_secs(),
                duration.subsec_nanos()
            );
        }
    }};
}
