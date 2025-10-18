#---------------------------------------------------------------------
# Makefile for KeyHunt
#
# Author : Jean-Luc PONS

SRC = Base58.cpp IntGroup.cpp Main.cpp Bloom.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      KeyHunt.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp hash/keccak160.cpp GmpUtil.cpp CmdParse.cpp

# Dynamic object and build directories based on CCAP value
CCAP ?= 86  # Default Compute Capability
OBJDIR = build/ccap_$(CCAP)/obj
BINDIR = build/ccap_$(CCAP)/bin

ifdef gpu

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o Bloom.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o KeyHunt.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o hash/keccak160.o \
        GPU/GPUEngine.o GmpUtil.o CmdParse.o)

else

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o Bloom.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o KeyHunt.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o hash/keccak160.o \
        GmpUtil.o CmdParse.o)

endif

CXX        = g++
CUDA       = /usr/local/cuda-13.0
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc

ifdef gpu
ifdef debug
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include -std=c++23
else
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include -std=c++23
endif
LFLAGS     = -lgmp -lpthread -L$(CUDA)/lib64 -lcudart -lssl -lcrypto -lsqlite3
else
ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include -std=c++23
else
CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include -std=c++23
endif
LFLAGS     = -lgmp -lpthread
endif

#---------------------------------------------------------------------

ifdef gpu
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu | $(OBJDIR)/GPU
ifdef debug
	$(NVCC) -G -g -maxrregcount=0 --ptxas-options=-v --extended-lambda --compile \
	--compiler-options -fPIC -ccbin $(CXXCUDA) -m64 \
	-I$(CUDA)/include -gencode=arch=compute_$(CCAP),code=sm_$(CCAP) \
	-o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --extended-lambda --compile \
	--compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 \
	-I$(CUDA)/include -gencode=arch=compute_$(CCAP),code=sm_$(CCAP) \
	-o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
endif

$(OBJDIR)/%.o : %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: $(BINDIR)/KeyHunt

$(BINDIR)/KeyHunt: $(OBJET) | $(BINDIR)
	@echo "Building KeyHunt..."
	$(CXX) $(OBJET) $(LFLAGS) -o $@

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	mkdir -p $(OBJDIR)/GPU

$(OBJDIR)/hash: $(OBJDIR)
	mkdir -p $(OBJDIR)/hash

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	@echo "Cleaning..."
	@rm -rf build/ccap_$(CCAP)

.PHONY: all clean
