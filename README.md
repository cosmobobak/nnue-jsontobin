# nnue-jsontobin

A little utility for converting from [Marlinflow](https://github.com/dsekercioglu/marlinflow)'s NNUE JSON format into a binary NNUE for embedding into chess engine executables.
Currently supports relative networks and HalfKA networks.

## Usage

Most typical usage is to convert a JSON file into a unified binary file, with an 8-bit output weight array.
This is achieved by running the following command:

```
nnue-jsontobin INPUT.json --output OUTPUT.bin
```

This will produce a binary file with the following structure:

in C++:
```cpp
struct alignas(64) NetworkWeights {
    std::array<std::int16_t, 768 * NEURONS * BUCKETS> feature_weights;
    std::array<std::int16_t, NEURONS>                 feature_biases;
    std::array<std::int8_t , NEURONS * 2>             output_weights;
    std::int16_t                                      output_bias;
}
```
or, in Rust:
```rust
#[repr(C, align(64))]
struct NetworkWeights {
    feature_weights: [i16; 768 * NEURONS * BUCKETS],
    feature_biases:  [i16; NEURONS],
    output_weights:  [i8; NEURONS * 2],
    output_bias:     i16,
}
```

Binary files will be aligned to 64 bytes (size is a multiple of 64 bytes), and so will be padded with trailing zeros if necessary.

For backwards compatibility, you can also produce a binary file with a 16-bit output weight array, by adding the `--big-out` flag.

`nnue-jsontobin` performs quantisation of the network weights during the conversion process, and by default uses 255 and 64 as the quantisation factors.

These factors are configurable with the `--qa` and `--qb` flags, respectively.

## Building

`nnue-jsontobin` is written in Rust, and can be built with `cargo build --release`.

## License

`nnue-jsontobin` is licensed under the MIT license.
