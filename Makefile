EXE = nnue-jsontobin
ifeq ($(OS),Windows_NT)
	NAME := $(EXE)-x86_64-win.exe
else
	NAME := $(EXE)-x86_64-linux
endif

rule:
	cargo rustc --release -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=$(NAME)