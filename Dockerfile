FROM debian:11 AS build

ARG TARGETPLATFORM
ARG GRPC_INSTALL_DIR=/opt/.grpc
ARG GRPC_VERSION=1.60.0
ARG ONNX_INSTALL_DIR=/opt/.onnx
ARG ONNX_VERSION=1.16.3

RUN apt update -y && \
	apt install -y cmake && \
	apt install -y build-essential autoconf libtool pkg-config git

RUN git clone --recurse-submodules -b "v${GRPC_VERSION}" --depth 1 --shallow-submodules https://github.com/grpc/grpc

RUN cd grpc && \
	mkdir -p "$GRPC_INSTALL_DIR" && \
	mkdir -p cmake/build && \
	cd cmake/build && \
	cmake -DgRPC_INSTALL=ON \
	  -DgRPC_BUILD_TESTS=OFF \
	  -DCMAKE_INSTALL_PREFIX=$GRPC_INSTALL_DIR \
		../..

RUN cd grpc && \
	mkdir -p cmake/build && \
	cd cmake/build && \
	make -j `nproc` && \
	make install

RUN apt-get install -y curl

RUN mkdir -p "$ONNX_INSTALL_DIR" && \
	if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
		curl -s -L "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz" | tar xvz --strip-components=1 -C "$ONNX_INSTALL_DIR" ; \
	elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
		curl -s -L "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz" | tar xvz --strip-components=1 -C "$ONNX_INSTALL_DIR" ; \
	fi

ADD . .

RUN mkdir -p cmake/build && \
	cd cmake/build && \
	cmake -DONNXRuntime_ROOT_DIR=$ONNX_INSTALL_DIR \
	  -DCMAKE_PREFIX_PATH=$GRPC_INSTALL_DIR \
		../..

RUN cd cmake/build && \
	make -j `nproc`

RUN cp -r "$ONNX_INSTALL_DIR" /onnxruntime

FROM gcr.io/distroless/cc-debian11

COPY --from=build /onnxruntime/lib/* /lib/
COPY --from=build --chmod=777 cmake/build/onnxruntime_server /bin/onnxruntime_server

ENTRYPOINT ["/bin/onnxruntime_server"]
