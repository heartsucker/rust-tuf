# rust-tuf

[![Travis build Status](https://travis-ci.org/heartsucker/rust-tuf.svg?branch=master)](https://travis-ci.org/heartsucker/rust-tuf) [![Appveyor build status](https://ci.appveyor.com/api/projects/status/kfyvpkdvn5ap7dqc?svg=true)](https://ci.appveyor.com/project/heartsucker/rust-tuf)

A Rust implementation of [The Update Framework (TUF)](https://theupdateframework.github.io/).

Full documentation is hosted at [docs.rs](https://docs.rs/crate/tuf).

## Warning: Beta Software

This is under active development and may not suitable for production use. Further,
the API is unstable and you should be prepared to refactor on even patch releases.

## Contributing

Please make all pull requests to the `develop` branch.

### Testing

`rust-tuf` uses [`tuf-test-vectors`](https://github.com/heartsucker/tuf-test-vectors)
to generate integration tests. When adding a complicated feature it may be
necessary for you to make a separate pull request to that repository to ensure
the required behaviors are sufficiently tested.

### Bugs

This project has a **full disclosure** policy on security related errors. Please
treat these errors like all other bugs and file a public issue. Errors communicated
via other channels will be immediately made public.

## Legal

### License

This work is dual licensed under the MIT and Apache-2.0 licenses.
See [LICENSE-MIT](./LICENSE-MIT) and [LICENSE-APACHE](./LICENSE-APACHE) for details.

### Cryptography Notice

This software includes and uses cryptographic software. Your current country may have
restrictions on the import, export, possession, or use of cryptographic software. Check
your country's relevant laws before using this in any way. See
[Wassenaar](http://www.wassenaar.org/) for more info.
