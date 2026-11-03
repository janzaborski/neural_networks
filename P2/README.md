# Hopfield networks

Implementation and analysis of Hopfield networks.

## Build

```bash
cmake --preset release
cmake --build --preset release
```

Debug build with sanitizers:

```bash
cmake --preset debug
cmake --build --preset debug
```

## Test

```bash
ctest --preset debug
```

## Run

```bash
./build/release/app/hopfield_app
```
