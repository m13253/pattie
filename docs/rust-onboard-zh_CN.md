Rust 上车指南
-------------

一些入门教程里没讲的小细节。

## 下载工具链

### Linux、macOS、WSL

1. 不需要使用 `apt`、`brew`、`dnf` 等工具提前安装任何软件
2. 打开 [https://rustup.rs](https://rustup.rs)
3. 把网页上的代码贴到终端中
4. 如果要安装日更版：
    1. 按 `2`
    2. 按回车
    3. 输入 `nightly`，回车
    4. 之后一路回车

### Windows (MSVC ABI)

1. 如果你还未安装过 Visual Studio，先去下载 [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 打开 [https://rustup.rs](https://rustup.rs)
3. 下载 `rustup-init.exe` 并运行
4. 如果要安装日更版：
    1. 输入 `2`
    2. 按回车
    3. 输入 `nightly`
    4. 之后一路回车

### Windows (MinGW ABI)

1. 不需要先下载 Visual Studio，但是编译出来的软件不能和 MSVC ABI 兼容
2. 打开 [https://rustup.rs](https://rustup.rs)
3. 下载 `rustup-init.exe` 并运行
4. 输入 `2`
5. 输入 `x86_64-pc-windows-gnu`
6. 如果要安装日更版，输入 `nightly`，否则直接回车
7. 之后一路回车

### 切换稳定版和日更版工具链

虽然会有缓存，但日更版每天需要重新下载。

```bash
$ rustup default stable   # 切换到稳定版
$ rustup default nightly  # 切换到日更版
$ rustup update           # 更新日更版，每天都能更
```

## 编辑器支持

### Visual Studio Code

**不推荐** 那个名字叫 [Rust](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust) 的插件，推荐另一个叫 [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer) 的插件。

### Sublime Text

推荐 [Rust Enhanced](https://github.com/rust-lang/rust-enhanced) 插件，但我不知道是否比 VS Code 的这个好用。

### Vim

[`Rust.vim`](https://github.com/rust-lang/rust.vim)、[`Syntastic`](https://github.com/vim-syntastic/syntastic)

## 编译单个文件

基本上用不到的技能。因为我们有 Cargo。

```bash
$ rustc -o output input.rs
```

## 使用 Cargo 管理软件包

```bash
$ cargo new hello-world  # 如果文件夹提前建立好了，就不需要加名字
$ cd hello-world         # 里面有个 Hello world
$ cargo build            # 编译，关闭优化
$ cargo run              # 编译并运行，关闭优化
$ cargo build --release  # 编译，打开优化
$ cargo run --release    # 编译并运行，打开优化
```

编译好的东西在 `./target` 里。此外，`cargo new` 会自动配置 Git。

## 如何找库

* 官方仓库：[https://crates.io](https://crates.io)
* 非官方排行榜（搜索不太准）：[https://lib.rs](https://lib.rs)
* LibHunt，有相似软件对比功能：[https://www.libhunt.com/l/rust](https://www.libhunt.com/l/rust)
* Google、社区、论坛

## 如何读文档

* DevDocs.io：[https://devdocs.io](https://devdocs.io)
    1. 点左上角搜索框旁边的菜单，选 Preferences
    2. 去掉所有语言，勾上 Rust
    3. 点左上角的 Apply
* 官方标准库文档：[https://doc.rust-lang.org/std/](https://doc.rust-lang.org/std/)
* 第三方库文档：[https://docs.rs](https://docs.rs)
* 从源代码编译出 HTML 版文档：

    ```bash
    $ cargo doc --no-deps
    ```

    编译好的文档会出现在 `./target/doc/包名/index.html`。

## 教程里没提到的小知识

### 函数式错误处理

三种错误：

1. `Option`，分为 `Some(value)` 和 `None`。
2. `Result`，分为 `Ok(value)` 和 `Err(error)`。
3. `panic!(reason)`，直接让程序崩溃。

处理办法：

1. 通过 `a.ok()` 或 `a.ok_or(error)` 在 `Result` 和 `Option` 之间转换。
2. 通过 `a.expect(reason)` 转换成 `panic!`。
3. 通过 `a?` 提取内容，若失败则自动 `return a`。
4. 通过 `if let` 或 `match` 拆内容。

实用工具库：

* [anyhow](https://lib.rs/crates/anyhow)

### 迭代器

任何数据类型，只要满足 [`std::iter::Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html) 的 trait，就是迭代器。

1. `iter()`, `iter_mut()`, `into_iter()`, …
2. `map`, `reduce`, `collect::<Vec<_>>()`, `filter`, `flat_map`, `fold`, `for_each`, `take`, `zip`, …
