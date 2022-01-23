// Adapted from https://github.com/seanmonstar/pretty-env-logger
// Original Copyright (c) 2017 Sean McArthur
// MIT / Apache-2.0 Licensed

use env_logger;
use env_logger::fmt::{Color, Style, StyledValue};
use log::{Level, LevelFilter};

pub fn init() {
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .filter_module("Event", LevelFilter::Trace)
        .format(|buf, record| {
            use std::io::Write;

            let target = record.target();

            let mut style = buf.style();
            let level = colored_level(&mut style, record.level());

            let mut style = buf.style();
            let target = style.set_bold(true).value(target);

            writeln!(buf, "[{} {}] {}", level, target, record.args())
        })
        .init();
}

fn colored_level<'a>(style: &'a mut Style, level: Level) -> StyledValue<'a, &'static str> {
    match level {
        Level::Trace => style.set_color(Color::Magenta).value("TRACE"),
        Level::Debug => style.set_color(Color::Blue).value("DEBUG"),
        Level::Info => style.set_color(Color::Green).value("INFO "),
        Level::Warn => style.set_color(Color::Yellow).value("WARN "),
        Level::Error => style.set_color(Color::Red).value("ERROR"),
    }
}
