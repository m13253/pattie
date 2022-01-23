use crossbeam_channel;
use crossbeam_utils;
use log::trace;
use scopeguard::defer;
use std::borrow::Cow;
use std::fs::File;
use std::io;
use std::mem;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const DEFAULT_EVENT_BUFFER_SIZE: usize = 256;
const DEFAULT_FILE_BUFFER_SIZE: usize = 8192;

/// A performance tracer
#[derive(Clone, Debug, Default)]
pub struct Tracer(Option<TracerInner>);

#[derive(Clone, Debug)]
struct TracerInner {
    tx: crossbeam_channel::Sender<Record>,
    _waiter: Arc<ThreadWaiter>,
}

/// An event for the performance tracer
#[derive(Copy, Clone, Debug)]
pub struct Event<'a>(Option<EventInner<'a>>);

#[derive(Copy, Clone, Debug)]
struct EventInner<'a> {
    start_time: Instant,
    tx: &'a crossbeam_channel::Sender<Record>,
}

/// An event for the performance tracer, but automatically finishes when dropped
#[derive(Clone, Debug)]
pub struct EventGuard<'a>(Option<EventGuardInner<'a>>);

#[derive(Clone, Debug)]
struct EventGuardInner<'a> {
    start_time: Instant,
    name: Cow<'static, str>,
    tx: &'a crossbeam_channel::Sender<Record>,
}

#[derive(Clone, Debug)]
struct Record {
    start_time: Instant,
    finish_time: Instant,
    name: Cow<'static, str>,
}

#[derive(Debug)]
struct ThreadWaiter {
    parker: crossbeam_utils::sync::Parker,
}

impl Tracer {
    /// Create a disabled tracer that does nothing.
    #[inline]
    pub fn new_dummy() -> Self {
        Self::default()
    }

    /// Create a tracer that records events to an already-open file.
    ///
    /// Returns error if thread creation fails.
    #[inline]
    pub fn new_to_file(file: File) -> Result<Self, io::Error> {
        let epoch = Instant::now();
        let (tx, rx) = crossbeam_channel::bounded(DEFAULT_EVENT_BUFFER_SIZE);
        let parker = crossbeam_utils::sync::Parker::new();
        let unparker = parker.unparker().clone();
        thread::Builder::new()
            .name("tracer".to_string())
            .spawn(move || Self::thread_main_with_file(rx, unparker, file, epoch))?;
        Ok(Self(Some(TracerInner {
            tx,
            _waiter: Arc::new(ThreadWaiter { parker }),
        })))
    }

    /// Create a tracer that records events to a file.
    ///
    /// Returns error if file or thread creation fails.
    #[inline]
    pub fn new_to_filename(filename: impl AsRef<Path>) -> Result<Self, io::Error> {
        if filename.as_ref().as_os_str() == "-" {
            return Self::new_to_stdout();
        }
        let epoch = Instant::now();
        let file = File::create(filename)?;
        let (tx, rx) = crossbeam_channel::bounded(DEFAULT_EVENT_BUFFER_SIZE);
        let parker = crossbeam_utils::sync::Parker::new();
        let unparker = parker.unparker().clone();
        thread::Builder::new()
            .name("tracer".to_string())
            .spawn(move || Self::thread_main_with_file(rx, unparker, file, epoch))?;
        Ok(Self(Some(TracerInner {
            tx,
            _waiter: Arc::new(ThreadWaiter { parker }),
        })))
    }

    /// Create a tracer that records events to the standard output.
    ///
    /// Returns error if thread creation fails.
    #[inline]
    pub fn new_to_stdout() -> Result<Self, io::Error> {
        let (tx, rx) = crossbeam_channel::bounded(DEFAULT_EVENT_BUFFER_SIZE);
        let parker = crossbeam_utils::sync::Parker::new();
        let unparker = parker.unparker().clone();
        thread::Builder::new()
            .name("tracer".to_string())
            .spawn(move || Self::thread_main_with_stdout(rx, unparker))?;
        Ok(Self(Some(TracerInner {
            tx,
            _waiter: Arc::new(ThreadWaiter { parker }),
        })))
    }

    /// Start a new event.
    ///
    /// If the tracer is disabled, this function does nothing.
    ///
    /// To record the duration of the event, call `finish()` when the event finishes.
    /// You can call `finish()` multiple times to record multiple events with same starting point.
    #[inline(always)]
    #[must_use]
    pub fn start(&self) -> Event {
        if let Some(TracerInner { ref tx, _waiter: _ }) = self.0 {
            let start_time = Instant::now();
            Event(Some(EventInner { start_time, tx }))
        } else {
            Event(None)
        }
    }

    /// Start a new event similar to `start()`, but automatically finishes when the variable's lifetime ends.
    #[inline(always)]
    #[must_use]
    pub fn start_until_drop(&self, name: impl Into<Cow<'static, str>>) -> EventGuard {
        if let Some(TracerInner { ref tx, _waiter: _ }) = self.0 {
            let start_time = Instant::now();
            EventGuard(Some(EventGuardInner {
                start_time,
                name: name.into(),
                tx,
            }))
        } else {
            EventGuard(None)
        }
    }

    fn thread_main_with_file(
        rx: crossbeam_channel::Receiver<Record>,
        finish: crossbeam_utils::sync::Unparker,
        mut file: File,
        epoch: Instant,
    ) {
        use std::io::Write;

        defer! {
            finish.unpark();
        }
        write!(
            file,
            "Event name,Start time (sec),Finish time (sec),Duration (sec)\r\n"
        )
        .unwrap();
        let mut file = io::BufWriter::with_capacity(DEFAULT_FILE_BUFFER_SIZE, file);

        for record in rx {
            let name = record.name.as_ref();
            let start = record.start_time.duration_since(epoch);
            let finish = record.finish_time.duration_since(epoch);
            let duration = record.finish_time.duration_since(record.start_time);

            if name.contains('"') {
                file.write_all(b"\"").unwrap();
                for b in name.bytes() {
                    if b == b'"' {
                        file.write_all(b"\"\"").unwrap();
                    } else {
                        file.write_all(&[b]).unwrap();
                    }
                }
                file.write_all(b"\"").unwrap();
            } else {
                file.write_all(name.as_bytes()).unwrap();
            }
            write!(
                file,
                ",{}.{:09},{}.{:09},{}.{:09}\r\n",
                start.as_secs(),
                start.subsec_nanos(),
                finish.as_secs(),
                finish.subsec_nanos(),
                duration.as_secs(),
                duration.subsec_nanos()
            )
            .unwrap();
        }
    }

    fn thread_main_with_stdout(
        rx: crossbeam_channel::Receiver<Record>,
        finish: crossbeam_utils::sync::Unparker,
    ) {
        defer! { finish.unpark(); }

        for record in rx {
            let duration = record.finish_time.duration_since(record.start_time);
            trace!(target: "Event", "({}) {}.{:09} seconds", record.name, duration.as_secs(), duration.subsec_nanos());
        }
    }
}

impl Event<'_> {
    /// Finish the event and record the duration.
    ///
    /// If the event is disabled, this function does nothing.
    #[inline(always)]
    pub fn finish(&self, name: impl Into<Cow<'static, str>>) {
        if let Some(EventInner { start_time, tx }) = self.0 {
            let finish_time = Instant::now();
            tx.send(Record {
                start_time,
                finish_time,
                name: name.into(),
            })
            .unwrap();
        }
    }
}

impl EventGuard<'_> {
    /// Finish the event early without waiting for it to drop and record the duration.
    ///
    /// If the event is disabled, this function does nothing.
    #[inline(always)]
    pub fn finish_early(self) {}
}

impl Drop for EventGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(EventGuardInner {
            start_time,
            ref mut name,
            tx,
        }) = self.0
        {
            let finish_time = Instant::now();
            tx.send(Record {
                start_time,
                finish_time,
                name: mem::take(name),
            })
            .unwrap();
        }
    }
}

impl Drop for ThreadWaiter {
    #[inline(always)]
    fn drop(&mut self) {
        self.parker.park();
    }
}
