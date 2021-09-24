Rust Onboard
------------

This document is mostly some details not covered in tutorials.

## Downloading the toolchain

### Linux, macOS, WSL

1. No need to install anything with `apt`, `brew`, `dnf`.
2. Navigate to [https://rustup.rs](https://rustup.rs)
3. Paste the command line from the web page into a terminal
4. If you want to install the nightly version:
    1. Type `2`
    2. Press Enter
    3. Type `nightly`, then press Enter
    4. Press Enter until complete

### Windows (MSVC ABI)

1. If you never installed Visual Studio, download [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.
2. Navigate to [https://rustup.rs](https://rustup.rs)
3. Download and run `rustup-init.exe`
4. If you want to install the nightly version:
    1. Type `2`
    2. Press Enter
    3. Type `nightly`, then press Enter
    4. Press Enter until complete

### Windows (MinGW ABI)

1. No need to download Visual Studio, but compiled binaries are incompatible with MSVC ABI
2. Navigate to [https://rustup.rs](https://rustup.rs)
3. Download and run `rustup-init.exe`
4. Type `2`
5. Type `x86_64-pc-windows-gnu`
6. If you also want nightly version, type `nightly`, otherwise press Enter
7. Press Enter until complete

### Switching between stable and nightly versions

Although the inactive version is cached locally, you need to re-download the nightly version every day.

```bash
$ rustup default stable   # Switch to stable version
$ rustup default nightly  # Switch to nightly version
$ rustup update           # Update the nightly version, available every day
```

## Editor integration

### Visual Studio Code

The extension named [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer) **rather than** [Rust](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust) is recommended.

### Sublime Text

[Rust Enhanced](https://github.com/rust-lang/rust-enhanced) is recommended. But I am not sure whether it is as helpful as VS Code.

### Vim

[`Rust.vim`](https://github.com/rust-lang/rust.vim), [`Syntastic`](https://github.com/vim-syntastic/syntastic)

## Compile single source file

We usually use Cargo, not this command.

```bash
$ rustc -o output input.rs
```

## Use Cargo to manage the package

```bash
$ cargo new hello-world  # You can omit the name if the folder is already made
$ cd hello-world         # There is a "Hello world" program inside
$ cargo build            # Build without optimization
$ cargo run              # Build then run, without optimization
$ cargo build --release  # Build with optimization
$ cargo run --release    # Build then run, with optimization
```

Built binaries are stored in the `./target` folder. Additionally, `cargo new` configures Git automatically.

## How to hunt for libraries

* Official repository: [https://crates.io](https://crates.io)
* Unofficial ranking (the search function is inaccurate): [https://lib.rs](https://lib.rs)
* LibHunt, with comparison feature: [https://www.libhunt.com/l/rust](https://www.libhunt.com/l/rust)
* Google, community, forums

## How to find documentations

* DevDocs.io: [https://devdocs.io](https://devdocs.io)
    1. Click the menu button at the top-left corner, next to the search box, choose Preferences
    2. Select only Rust
    3. Click Apply at the top-left corner
* Official standard library: [https://doc.rust-lang.org/std/](https://doc.rust-lang.org/std/)
* Third-party libraries: [https://docs.rs](https://docs.rs)
* Build HTML documentations from source code:

    ```bash
    $ cargo doc --no-deps
    ```

    The built documentation is located at `./target/doc/[crate name]/index.html`。

## Trivia absent in most tutorials

### Functional style error handing

Three types of errors:

1. `Option`, can be `Some(value)` or `None`.
2. `Result`, can be `Ok(value)` or `Err(error)`.
3. `panic!(reason)`, crashes the program immediately.

Error handling:

1. Use `a.ok()` or `a.ok_or(error)` to convert between `Result` and `Option`.
2. Use `a.expect(reason)` to convert into `panic!`.
3. Use `a?` to extract the value, which `return a` automatically on error.
4. Use `if let` or `match` to extract the value.

Utility libraries:

* [anyhow](https://lib.rs/crates/anyhow)

### Iterators

Any data structure, as long as it satisfies the [`std::iter::Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html) trait, can be iterated.

1. `iter()`, `iter_mut()`, `into_iter()`, …
2. `map`, `reduce`, `collect::<Vec<_>>()`, `filter`, `flat_map`, `fold`, `for_each`, `take`, `zip`, …
