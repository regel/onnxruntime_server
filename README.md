# How to Use ONNX Runtime Server for Prediction

ONNX Runtime Server provides an easy way to start an inferencing server for prediction with GRPC endpoints.

The CLI command to start the server is shown below:

```bash
$ ./onnxruntime_server --helpfull
onnxruntime_server: ./onnxruntime_server --model_path trained.pt

  Flags from onnxruntime_server.cpp:
    --address (The base server address); default: "0.0.0.0";
    --grpc_port (GRPC port to listen to requests); default: 50051;
    --log_level (Logging level. Allowed options (case sensitive): info, warning,
      error, fatal); default: INFO;
    --model_path (Path to ONNX model); default: ;
    --num_threads (Number of server threads); default: 0;
```

**Note**: The only mandatory argument for the program here is `model_path`

## Start the Server

To host an ONNX model as an inferencing server, simply run:

```
./onnxruntime_server --model_path /<your>/<model>/<path>
```

### Dependencies

The Abseil C++ library is cloned as a submodule. Run the following commands after cloning this repository:

```bash
git submodule init
git submodule update
```

## Build

Generate the Makefile:

```
% mkdir build && cd build
% cmake ..
```

Build the sources:

```
% make
...
[ 96%] Built target absl_random_internal_distribution_test_util
[ 97%] Built target absl_status
[ 98%] Built target absl_statusor
[ 99%] Built target absl_cordz_sample_token
[100%] Built target absl_bad_any_cast_impl
```

## Built With

* [Abseil](https://abseil.io/) - An open-source collection of C++ code (compliant to C++11) designed to augment the C++ standard library.

## GRPC Endpoint

To use the GRPC endpoint, the protobuf can be found [here](./proto/inference/inference.proto). You could generate your client and make a GRPC call to it. To learn more about how to generate the client code and call to the server, please refer to [the tutorials of GRPC](https://grpc.io/docs/tutorials/).

## Advanced Topics

### Number of Worker Threads

You can change this to optimize server utilization. The default is the number of CPU cores on the host machine.

## Extensions

The following Visual Studio Code extensions are highly recommended for working with this project:

* [C/C++ for Visual Studio Code](https://github.com/microsoft/vscode-cpptools) - Provides rich C and C++ language support, including features such as IntelliSense, debugging, and code navigation.
* CMake For VisualStudio Code - Enables convenient configuration and building of CMake projects within VS Code.
* CMake Tools - Provides additional CMake support, including capabilities for configuring, building, and testing CMake projects.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
