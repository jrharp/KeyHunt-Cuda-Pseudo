#include "KeyHunt.h"
#include "GmpUtil.h"
#include "Base58.h"
#include "hash/sha256.h"
#include "hash/keccak160.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <inttypes.h>
#include <limits>
#include <cstdio>
#ifndef WIN64
#include <pthread.h>
#endif

//using namespace std;

Point Gn[CPU_GRP_SIZE / 2];
Point _2Gn;

// ----------------------------------------------------------------------------

KeyHunt::KeyHunt(const std::string& inputFile, int compMode, int searchMode, int coinType, bool useGpu,
        const std::string& outputFile, bool useSSE, uint32_t maxFound, uint64_t rKey,
        const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit, int gpuStepMultiplier)
{
        this->compMode = compMode;
        this->useGpu = useGpu;
        this->outputFile = outputFile;
        this->useSSE = useSSE;
	this->nbGPUThread = 0;
	this->inputFile = inputFile;
	this->maxFound = maxFound;
	this->rKey = rKey;
	this->searchMode = searchMode;
        this->coinType = coinType;
        this->rangeStart.SetBase16(rangeStart.c_str());
        this->initialRangeStart.Set(&this->rangeStart);
        this->rangeEnd.SetBase16(rangeEnd.c_str());
        this->rangeDiff2.Set(&this->rangeEnd);
        this->rangeDiff2.Sub(&this->rangeStart);
        this->lastrKey = 0;
        this->gpuStepMultiplierRequested = std::max(1, gpuStepMultiplier);

        secp = new Secp256K1();
        secp->Init();

	// load file
	FILE* wfd;
	uint64_t N = 0;

	wfd = fopen(this->inputFile.c_str(), "rb");
	if (!wfd) {
		printf("%s can not open\n", this->inputFile.c_str());
		exit(1);
	}

#ifdef WIN64
	_fseeki64(wfd, 0, SEEK_END);
	N = _ftelli64(wfd);
#else
	fseek(wfd, 0, SEEK_END);
	N = ftell(wfd);
#endif

	int K_LENGTH = 20;
	if (this->searchMode == (int)SEARCH_MODE_MX)
		K_LENGTH = 32;

	N = N / K_LENGTH;
	rewind(wfd);

	DATA = (uint8_t*)malloc(N * K_LENGTH);
	memset(DATA, 0, N * K_LENGTH);

	uint8_t* buf = (uint8_t*)malloc(K_LENGTH);;

	bloom = new Bloom(2 * N, 0.000001);

	uint64_t percent = (N - 1) / 100;
	uint64_t i = 0;
	printf("\n");
	while (i < N && !should_exit) {
		memset(buf, 0, K_LENGTH);
		memset(DATA + (i * K_LENGTH), 0, K_LENGTH);
		if (fread(buf, 1, K_LENGTH, wfd) == K_LENGTH) {
			bloom->add(buf, K_LENGTH);
			memcpy(DATA + (i * K_LENGTH), buf, K_LENGTH);
			if ((percent != 0) && i % percent == 0) {
				printf("\rLoading      : %" PRIu64 " %%", (i / percent));
				fflush(stdout);
			}
		}
		i++;
	}
	fclose(wfd);
	free(buf);

	if (should_exit) {
		delete secp;
		delete bloom;
		if (DATA)
			free(DATA);
		exit(0);
	}

	BLOOM_N = bloom->get_bytes();
	TOTAL_COUNT = N;
	targetCounter = i;
	if (coinType == COIN_BTC) {
		if (searchMode == (int)SEARCH_MODE_MA)
			printf("Loaded       : %s Bitcoin addresses\n", formatThousands(i).c_str());
		else if (searchMode == (int)SEARCH_MODE_MX)
			printf("Loaded       : %s Bitcoin xpoints\n", formatThousands(i).c_str());
	}
	else {
		printf("Loaded       : %s Ethereum addresses\n", formatThousands(i).c_str());
	}

	printf("\n");

        bloom->print();
        printf("\n");

        initializePseudoRandomState();
        InitGenratorTable();

}

// ----------------------------------------------------------------------------

KeyHunt::KeyHunt(const std::vector<unsigned char>& hashORxpoint, int compMode, int searchMode, int coinType,
        bool useGpu, const std::string& outputFile, bool useSSE, uint32_t maxFound, uint64_t rKey,
        const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit, int gpuStepMultiplier)
{
        this->compMode = compMode;
        this->useGpu = useGpu;
        this->outputFile = outputFile;
        this->useSSE = useSSE;
	this->nbGPUThread = 0;
	this->maxFound = maxFound;
	this->rKey = rKey;
        this->searchMode = searchMode;
        this->coinType = coinType;
        this->rangeStart.SetBase16(rangeStart.c_str());
        this->initialRangeStart.Set(&this->rangeStart);
        this->rangeEnd.SetBase16(rangeEnd.c_str());
        this->rangeDiff2.Set(&this->rangeEnd);
        this->rangeDiff2.Sub(&this->rangeStart);
        this->targetCounter = 1;
        this->gpuStepMultiplierRequested = std::max(1, gpuStepMultiplier);

        secp = new Secp256K1();
        secp->Init();

	if (this->searchMode == (int)SEARCH_MODE_SA) {
		assert(hashORxpoint.size() == 20);
		for (size_t i = 0; i < hashORxpoint.size(); i++) {
			((uint8_t*)hash160Keccak)[i] = hashORxpoint.at(i);
		}
	}
	else if (this->searchMode == (int)SEARCH_MODE_SX) {
		assert(hashORxpoint.size() == 32);
		for (size_t i = 0; i < hashORxpoint.size(); i++) {
			((uint8_t*)xpoint)[i] = hashORxpoint.at(i);
		}
	}
        printf("\n");

        initializePseudoRandomState();
        InitGenratorTable();
}

// ----------------------------------------------------------------------------

void KeyHunt::InitGenratorTable()
{
        // Compute Generator table G[n] = (n+1)*G
        int halfGroup = cpuGroupSize / 2;
        Point g = secp->G;
        if (halfGroup > 0) {
                Gn[0] = g;
        }

        if (halfGroup > 1) {
                g = secp->DoubleDirect(g);
                Gn[1] = g;
                for (int i = 2; i < halfGroup; i++) {
                        g = secp->AddDirect(g, secp->G);
                        Gn[i] = g;
                }
        }

        // _2Gn = cpuGroupSize*G
        if (halfGroup > 0) {
                _2Gn = secp->DoubleDirect(Gn[halfGroup - 1]);
        }
        else {
                _2Gn = secp->DoubleDirect(secp->G);
        }

	char* ctimeBuff;
	time_t now = time(NULL);
	ctimeBuff = ctime(&now);
	printf("Start Time   : %s", ctimeBuff);

	if (rKey > 0) {
		printf("Base Key     : Randomly changes on every %" PRIu64 " Mkeys\n", rKey);
	}
	printf("Global start : %s (%d bit)\n", this->rangeStart.GetBase16().c_str(), this->rangeStart.GetBitLength());
	printf("Global end   : %s (%d bit)\n", this->rangeEnd.GetBase16().c_str(), this->rangeEnd.GetBitLength());
        printf("Global range : %s (%d bit)\n", this->rangeDiff2.GetBase16().c_str(), this->rangeDiff2.GetBitLength());

}

// ----------------------------------------------------------------------------

bool KeyHunt::loadPseudoRandomState(uint64_t& resumeIndex) const
{
        if (pseudoState.stateFile.empty())
                return false;

        std::ifstream in(pseudoState.stateFile);
        if (!in.good())
                return false;

        uint64_t value = 0;
        in >> value;
        if (!in.good())
                return false;

        resumeIndex = value;
        return true;
}

// ----------------------------------------------------------------------------

void KeyHunt::persistPseudoRandomState(uint64_t completedCount)
{
        if (!pseudoRandomEnabled || pseudoState.stateFile.empty())
                return;

        if (pseudoState.lastPersisted == completedCount)
                return;

        std::lock_guard<std::mutex> guard(pseudoState.fileMutex);
        std::ofstream out(pseudoState.stateFile, std::ios::trunc);
        if (!out.is_open()) {
                if (!pseudoState.persistWarningShown) {
                        printf("Warning: unable to persist pseudo-random state to %s\n", pseudoState.stateFile.c_str());
                        pseudoState.persistWarningShown = true;
                }
                return;
        }

        out << completedCount << '\n';
        if (!out.good()) {
                if (!pseudoState.persistWarningShown) {
                        printf("Warning: unable to persist pseudo-random state to %s\n", pseudoState.stateFile.c_str());
                        pseudoState.persistWarningShown = true;
                }
                return;
        }

        pseudoState.lastPersisted = completedCount;
        pseudoState.persistWarningShown = false;
}

// ----------------------------------------------------------------------------

uint64_t KeyHunt::permuteBlockIndex(uint64_t value) const
{
        uint64_t mask = pseudoState.blockMask;
        uint64_t x = value & mask;
        x ^= (x >> 12);
        x &= mask;
        x ^= ((x << 25) & mask);
        x ^= (x >> 27);
        x &= mask;
        x = (x * 2685821657736338717ULL) & mask;
        x ^= (x >> 33);
        x &= mask;
        return x;
}

// ----------------------------------------------------------------------------

bool KeyHunt::acquirePseudoRandomBlock(Int& key, Point& startP, uint64_t& sequentialIndex, bool forGpu)
{
        if (!pseudoRandomEnabled)
                return false;

        uint64_t blockSize = pseudoState.blockKeyCount;
        if (blockSize == 0) {
                blockSize = static_cast<uint64_t>(cpuGroupSize);
        }

        while (true) {
                uint64_t idx = pseudoState.nextCounter.fetch_add(1);
                if (idx >= pseudoState.totalBlocks)
                        return false;

                uint64_t blockIndex = permuteBlockIndex(idx);
                sequentialIndex = idx;

                Int offset(blockIndex);
                offset.Mult(blockSize);
                key = initialRangeStart;
                key.Add(&offset);

                uint64_t midpoint = blockSize / 2;
                if (forGpu) {
                        const uint64_t compiledMidpoint = static_cast<uint64_t>(GPUEngine::GetCompiledGroupSize()) / 2ULL;
                        midpoint = std::min(midpoint, compiledMidpoint);
                }

                Int km(&key);
                km.Add(midpoint);
                startP = secp->ComputePublicKey(&km);
                return true;
        }
}

// ----------------------------------------------------------------------------

void KeyHunt::notifyPseudoRandomBlockComplete(uint64_t sequentialIndex)
{
        if (!pseudoRandomEnabled)
                return;

        uint64_t nextPersist = std::numeric_limits<uint64_t>::max();
        {
                std::lock_guard<std::mutex> guard(pseudoState.progressMutex);
                if (sequentialIndex >= pseudoState.totalBlocks)
                        return;

                pseudoState.completedBlocks.insert(sequentialIndex);
                uint64_t candidate = pseudoState.lowestUnpersisted;
                bool advanced = false;
                while (pseudoState.completedBlocks.count(candidate) > 0) {
                        pseudoState.completedBlocks.erase(candidate);
                        candidate++;
                        advanced = true;
                }

                if (advanced) {
                        pseudoState.lowestUnpersisted = candidate;
                        nextPersist = candidate;
                }
        }

        if (nextPersist != std::numeric_limits<uint64_t>::max()) {
                persistPseudoRandomState(nextPersist);
        }
}

void KeyHunt::startPseudoRandomGpuPrefetch(int targetQueueSize)
{
        if (!pseudoRandomEnabled || targetQueueSize <= 0)
                return;

        stopPseudoRandomGpuPrefetch();

        const size_t minimumCapacity = std::max<int>(CPU_GRP_SIZE, 1);
        {
                std::lock_guard<std::mutex> lock(pseudoGpuMutex);
                pseudoGpuQueue.clear();
                pseudoGpuQueueLimit = std::max<size_t>(static_cast<size_t>(targetQueueSize) * 2u, minimumCapacity);
                pseudoGpuStop.store(false, std::memory_order_relaxed);
        }

        unsigned int workerCount = std::thread::hardware_concurrency();
        if (workerCount == 0)
                workerCount = 2;
        workerCount = std::min<unsigned int>(workerCount, 8u);
        pseudoGpuActiveWorkers.store(static_cast<int>(workerCount));
        pseudoGpuWorkers.reserve(workerCount);
        for (unsigned int i = 0; i < workerCount; ++i) {
                pseudoGpuWorkers.emplace_back(&KeyHunt::pseudoRandomGpuWorker, this);
        }
}

void KeyHunt::stopPseudoRandomGpuPrefetch()
{
        pseudoGpuStop.store(true, std::memory_order_relaxed);
        pseudoGpuCv.notify_all();

        for (auto& worker : pseudoGpuWorkers) {
                if (worker.joinable()) {
                        worker.join();
                }
        }
        pseudoGpuWorkers.clear();
        pseudoGpuActiveWorkers.store(0);

        {
                std::lock_guard<std::mutex> lock(pseudoGpuMutex);
                pseudoGpuQueue.clear();
                pseudoGpuQueueLimit = 0;
                pseudoGpuStop.store(false, std::memory_order_relaxed);
        }
}

bool KeyHunt::dequeuePseudoRandomGpuBlock(PseudoRandomBlock& block)
{
        std::unique_lock<std::mutex> lock(pseudoGpuMutex);
        pseudoGpuCv.wait(lock, [this]() {
                return !pseudoGpuQueue.empty() || pseudoGpuActiveWorkers.load() == 0 || pseudoGpuStop.load();
        });

        if (!pseudoGpuQueue.empty()) {
                block = pseudoGpuQueue.front();
                pseudoGpuQueue.pop_front();
                lock.unlock();
                pseudoGpuCv.notify_all();
                return true;
        }

        return false;
}

void KeyHunt::pseudoRandomGpuWorker()
{
        while (true) {
                if (endOfSearch || pseudoGpuStop.load())
                        break;

                Int key;
                Point start;
                uint64_t sequential = 0;
                if (!acquirePseudoRandomBlock(key, start, sequential, true)) {
                        break;
                }

                std::unique_lock<std::mutex> lock(pseudoGpuMutex);
                pseudoGpuCv.wait(lock, [this]() {
                        return pseudoGpuStop.load() || pseudoGpuQueue.size() < pseudoGpuQueueLimit;
                });

                if (pseudoGpuStop.load()) {
                        break;
                }

                pseudoGpuQueue.emplace_back();
                auto& entry = pseudoGpuQueue.back();
                entry.key = key;
                entry.startPoint = start;
                entry.sequentialIndex = sequential;
                lock.unlock();
                pseudoGpuCv.notify_all();
        }

        pseudoGpuActiveWorkers.fetch_sub(1);
        pseudoGpuCv.notify_all();
}

// ----------------------------------------------------------------------------

void KeyHunt::initializePseudoRandomState()
{
        pseudoRandomEnabled = false;
        pseudoRandomCpuEnabled = false;
        cpuGroupSize = CPU_GRP_SIZE;
        gpuStepMultiplierEffective = useGpu ? gpuStepMultiplierRequested : 1;
        pseudoState.totalKeys = 0;
        pseudoState.totalBlocks = 0;
        pseudoState.blockMask = 0;
        pseudoState.blockBits = 0;
        pseudoState.blockKeyCount = 0;
        pseudoState.nextCounter.store(0);
        pseudoState.stateFile.clear();
        pseudoState.lastPersisted = std::numeric_limits<uint64_t>::max();
        pseudoState.persistWarningShown = false;
        pseudoState.lowestUnpersisted = 0;
        pseudoState.completedBlocks.clear();

        if (searchMode != (int)SEARCH_MODE_SA || coinType != COIN_BTC)
                return;

        if (rangeDiff2.GetBitLength() > 128) {
                printf("Pseudo-random traversal disabled: range span exceeds 128 bits.\n");
                return;
        }

        Int inclusiveRange(&rangeDiff2);
        inclusiveRange.AddOne();
        if (inclusiveRange.IsZero())
                return;

        const bool inclusiveRangeFits64 = (inclusiveRange.GetSize64() <= 1);
        const uint64_t inclusiveRangeLow = inclusiveRangeFits64 ? inclusiveRange.bits64[0] : 0;

        uint64_t targetGroupSize = static_cast<uint64_t>(CPU_GRP_SIZE);
#ifdef WITHGPU
        if (useGpu) {
                const uint64_t gpuTarget = static_cast<uint64_t>(STEP_SIZE) * static_cast<uint64_t>(gpuStepMultiplierRequested);
                targetGroupSize = std::max<uint64_t>(targetGroupSize, gpuTarget);
        }
#endif
        uint64_t cappedGroup = targetGroupSize;
        Int targetGroupInt(targetGroupSize);
        if (inclusiveRange.IsLowerOrEqual(&targetGroupInt)) {
                if (!inclusiveRangeFits64) {
                        printf("Pseudo-random traversal disabled: unable to represent the range with current limits.\n");
                        return;
                }
                cappedGroup = inclusiveRangeLow;
        }

        uint64_t candidateGroup = 1;
        while ((candidateGroup << 1) <= cappedGroup) {
                candidateGroup <<= 1;
        }

        const uint64_t minGroupSize = 4;
        if (candidateGroup < minGroupSize) {
                printf("Pseudo-random traversal disabled: range too small for pseudo-random traversal.\n");
                return;
        }

        auto countTrailingZeros = [](uint64_t value) {
                unsigned int zeros = 0;
                while ((value & 1ULL) == 0ULL) {
                        zeros++;
                        value >>= 1;
                }
                return zeros;
        };

        auto isMultipleOfPowerOfTwo = [&](uint64_t group) {
                if (group == 0 || (group & (group - 1)) != 0)
                        return false;

                unsigned int zeroBits = countTrailingZeros(group);
                if (zeroBits >= NB64BLOCK * 64)
                        return false;

                unsigned int wholeLimbs = zeroBits / 64;
                for (unsigned int i = 0; i < wholeLimbs; ++i) {
                        if (inclusiveRange.bits64[i] != 0)
                                return false;
                }

                unsigned int offset = zeroBits % 64;
                if (offset != 0) {
                        uint64_t mask = (1ULL << offset) - 1ULL;
                        if ((inclusiveRange.bits64[wholeLimbs] & mask) != 0)
                                return false;
                }

                return true;
        };

        bool totalBlocksOverflowed = false;

        auto computeTotalBlocks = [&](uint64_t group, uint64_t& totalBlocksOut) {
                unsigned int zeroBits = countTrailingZeros(group);
                Int quotient(&inclusiveRange);
                quotient.ShiftR(zeroBits);

                if (quotient.GetSize64() > 1) {
                        totalBlocksOverflowed = true;
                        return false;
                }

                totalBlocksOut = quotient.bits64[0];
                return true;
        };

        bool foundGroup = false;
        uint64_t totalBlocks = 0;
        uint64_t workingGroup = candidateGroup;
        while (workingGroup >= minGroupSize) {
                if (isMultipleOfPowerOfTwo(workingGroup) && computeTotalBlocks(workingGroup, totalBlocks)) {
                        if (totalBlocks != 0 && (totalBlocks & (totalBlocks - 1)) == 0) {
                                foundGroup = true;
                                break;
                        }
                }
                workingGroup >>= 1;
        }

        if (!foundGroup) {
                if (totalBlocksOverflowed) {
                        printf("Pseudo-random traversal disabled: range span requires more than 2^64 blocks.\n");
                }
                else {
                        printf("Pseudo-random traversal disabled: unable to derive a suitable group size for the range.\n");
                }
                return;
        }

        pseudoState.blockKeyCount = workingGroup;
        if (pseudoState.blockKeyCount == 0) {
                gpuStepMultiplierEffective = useGpu ? gpuStepMultiplierRequested : 1;
                return;
        }

        if (pseudoState.blockKeyCount <= static_cast<uint64_t>(CPU_GRP_SIZE)) {
                cpuGroupSize = static_cast<int>(pseudoState.blockKeyCount);
                pseudoRandomCpuEnabled = true;
        }
        else {
                cpuGroupSize = CPU_GRP_SIZE;
                pseudoRandomCpuEnabled = false;
        }

        if (useGpu) {
                if (pseudoState.blockKeyCount >= static_cast<uint64_t>(STEP_SIZE)) {
                        uint64_t multiplier = pseudoState.blockKeyCount / static_cast<uint64_t>(STEP_SIZE);
                        if (multiplier == 0) {
                                multiplier = 1;
                        }
                        if (multiplier > static_cast<uint64_t>(gpuStepMultiplierRequested)) {
                                multiplier = static_cast<uint64_t>(gpuStepMultiplierRequested);
                        }
                        gpuStepMultiplierEffective = static_cast<int>(std::max<uint64_t>(1, multiplier));
                }
                else {
                        gpuStepMultiplierEffective = 1;
                }
        }
        else {
                gpuStepMultiplierEffective = 1;
        }
        pseudoState.totalKeys = inclusiveRangeFits64 ? inclusiveRangeLow : std::numeric_limits<uint64_t>::max();
        pseudoState.totalBlocks = totalBlocks;
        pseudoState.blockMask = totalBlocks - 1;

        uint64_t tmp = totalBlocks;
        unsigned int bits = 0;
        while (tmp > 1) {
                bits++;
                tmp >>= 1;
        }
        pseudoState.blockBits = bits;

        std::string startHex = initialRangeStart.GetBase16();
        std::string endHex = rangeEnd.GetBase16();
        std::string key = startHex + ":" + endHex;
        std::size_t hashValue = std::hash<std::string>{}(key);
        char fileName[64];
        std::snprintf(fileName, sizeof(fileName), "keyhunt_sa_%016zx.state", hashValue);
        pseudoState.stateFile = fileName;

        uint64_t resumeIndex = 0;
        if (loadPseudoRandomState(resumeIndex)) {
                if (resumeIndex >= pseudoState.totalBlocks)
                        resumeIndex = resumeIndex % pseudoState.totalBlocks;
                pseudoState.nextCounter.store(resumeIndex);
                pseudoState.lastPersisted = resumeIndex;
                pseudoState.lowestUnpersisted = resumeIndex;
                printf("Pseudo-random traversal resume index: %" PRIu64 "\n", resumeIndex);
        }

        pseudoRandomEnabled = true;
        printf("Pseudo-random traversal enabled (%" PRIu64 " blocks, block size %" PRIu64 "). State file: %s\n",
                pseudoState.totalBlocks, pseudoState.blockKeyCount, pseudoState.stateFile.c_str());
        if (rKey > 0) {
                printf("Note: random base key refresh is disabled while pseudo-random traversal is active.\n");
        }
}

// ----------------------------------------------------------------------------

KeyHunt::~KeyHunt()
{
        delete secp;
        if (searchMode == (int)SEARCH_MODE_MA || searchMode == (int)SEARCH_MODE_MX)
		delete bloom;
	if (DATA)
		free(DATA);
}

// ----------------------------------------------------------------------------

double log1(double x)
{
	// Use taylor series to approximate log(1-x)
	return -x - (x * x) / 2.0 - (x * x * x) / 3.0 - (x * x * x * x) / 4.0;
}

void KeyHunt::output(std::string addr, std::string pAddr, std::string pAddrHex, std::string pubKey)
{

#ifdef WIN64
	WaitForSingleObject(ghMutex, INFINITE);
#else
	pthread_mutex_lock(&ghMutex);
#endif

	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			printf("Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}

	if (!needToClose)
		printf("\n");

	fprintf(f, "PubAddress: %s\n", addr.c_str());
	fprintf(stdout, "\n=================================================================================\n");
	fprintf(stdout, "PubAddress: %s\n", addr.c_str());

	if (coinType == COIN_BTC) {
		fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
	}

	fprintf(f, "Priv (HEX): %s\n", pAddrHex.c_str());
	fprintf(stdout, "Priv (HEX): %s\n", pAddrHex.c_str());

	fprintf(f, "PubK (HEX): %s\n", pubKey.c_str());
	fprintf(stdout, "PubK (HEX): %s\n", pubKey.c_str());

	fprintf(f, "=================================================================================\n");
	fprintf(stdout, "=================================================================================\n");

	if (needToClose)
		fclose(f);

#ifdef WIN64
	ReleaseMutex(ghMutex);
#else
	pthread_mutex_unlock(&ghMutex);
#endif

}

// ----------------------------------------------------------------------------

bool KeyHunt::checkPrivKey(std::string addr, Int& key, uint32_t incr, bool mode)
{
	Int k(&key), k2(&key);
	k.Add((uint64_t)incr);
	k2.Add((uint64_t)incr);
	// Check addresses
	Point p = secp->ComputePublicKey(&k);
	std::string px = p.x.GetBase16();
	std::string chkAddr = secp->GetAddress(mode, p);
	if (chkAddr != addr) {
		//Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		std::string chkAddr = secp->GetAddress(mode, p);
		if (chkAddr != addr) {
			printf("\n=================================================================================\n");
			printf("Warning, wrong private key generated !\n");
			printf("  PivK :%s\n", k2.GetBase16().c_str());
			printf("  Addr :%s\n", addr.c_str());
			printf("  PubX :%s\n", px.c_str());
			printf("  PivK :%s\n", k.GetBase16().c_str());
			printf("  Check:%s\n", chkAddr.c_str());
			printf("  PubX :%s\n", p.x.GetBase16().c_str());
			printf("=================================================================================\n");
			return false;
		}
	}
	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16(), secp->GetPublicKeyHex(mode, p));
	return true;
}

bool KeyHunt::checkPrivKeyETH(std::string addr, Int& key, uint32_t incr)
{
	Int k(&key), k2(&key);
	k.Add((uint64_t)incr);
	k2.Add((uint64_t)incr);
	// Check addresses
	Point p = secp->ComputePublicKey(&k);
	std::string px = p.x.GetBase16();
	std::string chkAddr = secp->GetAddressETH(p);
	if (chkAddr != addr) {
		//Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		std::string chkAddr = secp->GetAddressETH(p);
		if (chkAddr != addr) {
			printf("\n=================================================================================\n");
			printf("Warning, wrong private key generated !\n");
			printf("  PivK :%s\n", k2.GetBase16().c_str());
			printf("  Addr :%s\n", addr.c_str());
			printf("  PubX :%s\n", px.c_str());
			printf("  PivK :%s\n", k.GetBase16().c_str());
			printf("  Check:%s\n", chkAddr.c_str());
			printf("  PubX :%s\n", p.x.GetBase16().c_str());
			printf("=================================================================================\n");
			return false;
		}
	}
	output(addr, k.GetBase16()/*secp->GetPrivAddressETH(k)*/, k.GetBase16(), secp->GetPublicKeyHexETH(p));
	return true;
}

bool KeyHunt::checkPrivKeyX(Int& key, uint32_t incr, bool mode)
{
	Int k(&key);
	k.Add((uint64_t)incr);
	Point p = secp->ComputePublicKey(&k);
	std::string addr = secp->GetAddress(mode, p);
	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16(), secp->GetPublicKeyHex(mode, p));
	return true;
}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKeyCPU(LPVOID lpParam)
{
#else
void* _FindKeyCPU(void* lpParam)
{
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyCPU(p);
	return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam)
{
#else
void* _FindKeyGPU(void* lpParam)
{
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

// ----------------------------------------------------------------------------

void KeyHunt::checkMultiAddresses(bool compressed, Int key, int i, Point p1)
{
	unsigned char h0[20];

	// Point
	secp->GetHash160(compressed, p1, h0);
	if (CheckBloomBinary(h0, 20) > 0) {
		std::string addr = secp->GetAddress(compressed, h0);
		if (checkPrivKey(addr, key, i, compressed)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkMultiAddressesETH(Int key, int i, Point p1)
{
	unsigned char h0[20];

	// Point
	secp->GetHashETH(p1, h0);
	if (CheckBloomBinary(h0, 20) > 0) {
		std::string addr = secp->GetAddressETH(h0);
		if (checkPrivKeyETH(addr, key, i)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkSingleAddress(bool compressed, Int key, int i, Point p1)
{
	unsigned char h0[20];

	// Point
	secp->GetHash160(compressed, p1, h0);
	if (MatchHash((uint32_t*)h0)) {
		std::string addr = secp->GetAddress(compressed, h0);
		if (checkPrivKey(addr, key, i, compressed)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkSingleAddressETH(Int key, int i, Point p1)
{
	unsigned char h0[20];

	// Point
	secp->GetHashETH(p1, h0);
	if (MatchHash((uint32_t*)h0)) {
		std::string addr = secp->GetAddressETH(h0);
		if (checkPrivKeyETH(addr, key, i)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkMultiXPoints(bool compressed, Int key, int i, Point p1)
{
	unsigned char h0[32];

	// Point
	secp->GetXBytes(compressed, p1, h0);
	if (CheckBloomBinary(h0, 32) > 0) {
		if (checkPrivKeyX(key, i, compressed)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkSingleXPoint(bool compressed, Int key, int i, Point p1)
{
	unsigned char h0[32];

	// Point
	secp->GetXBytes(compressed, p1, h0);
	if (MatchXPoint((uint32_t*)h0)) {
		if (checkPrivKeyX(key, i, compressed)) {
			nbFoundKey++;
		}
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::checkMultiAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4)
{
	unsigned char h0[20];
	unsigned char h1[20];
	unsigned char h2[20];
	unsigned char h3[20];

	// Point -------------------------------------------------------------------------
	secp->GetHash160(compressed, p1, p2, p3, p4, h0, h1, h2, h3);
	if (CheckBloomBinary(h0, 20) > 0) {
		std::string addr = secp->GetAddress(compressed, h0);
		if (checkPrivKey(addr, key, i + 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h1, 20) > 0) {
		std::string addr = secp->GetAddress(compressed, h1);
		if (checkPrivKey(addr, key, i + 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h2, 20) > 0) {
		std::string addr = secp->GetAddress(compressed, h2);
		if (checkPrivKey(addr, key, i + 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (CheckBloomBinary(h3, 20) > 0) {
		std::string addr = secp->GetAddress(compressed, h3);
		if (checkPrivKey(addr, key, i + 3, compressed)) {
			nbFoundKey++;
		}
	}

}

// ----------------------------------------------------------------------------

void KeyHunt::checkSingleAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4)
{
	unsigned char h0[20];
	unsigned char h1[20];
	unsigned char h2[20];
	unsigned char h3[20];

	// Point -------------------------------------------------------------------------
	secp->GetHash160(compressed, p1, p2, p3, p4, h0, h1, h2, h3);
	if (MatchHash((uint32_t*)h0)) {
		std::string addr = secp->GetAddress(compressed, h0);
		if (checkPrivKey(addr, key, i + 0, compressed)) {
			nbFoundKey++;
		}
	}
	if (MatchHash((uint32_t*)h1)) {
		std::string addr = secp->GetAddress(compressed, h1);
		if (checkPrivKey(addr, key, i + 1, compressed)) {
			nbFoundKey++;
		}
	}
	if (MatchHash((uint32_t*)h2)) {
		std::string addr = secp->GetAddress(compressed, h2);
		if (checkPrivKey(addr, key, i + 2, compressed)) {
			nbFoundKey++;
		}
	}
	if (MatchHash((uint32_t*)h3)) {
		std::string addr = secp->GetAddress(compressed, h3);
		if (checkPrivKey(addr, key, i + 3, compressed)) {
			nbFoundKey++;
		}
	}

}

// ----------------------------------------------------------------------------

void KeyHunt::getCPUStartingKey(Int & tRangeStart, Int & tRangeEnd, Int & key, Point & startP)
{
	if (rKey <= 0) {
		key.Set(&tRangeStart);
	}
	else {
		key.Rand(&tRangeEnd);
	}
        Int km(&key);
        km.Add(static_cast<uint64_t>(cpuGroupSize) / 2);
	startP = secp->ComputePublicKey(&km);

}

// ----------------------------------------------------------------------------

void KeyHunt::FindKeyCPU(TH_PARAM * ph)
{

	// Global init
	int thId = ph->threadId;
	Int tRangeStart = ph->rangeStart;
	Int tRangeEnd = ph->rangeEnd;
	counters[thId] = 0;

	// CPU Thread
        IntGroup* grp = new IntGroup(cpuGroupSize / 2 + 1);

	// Group Init
        Int key;// = new Int();
        Point startP;// = new Point();
        if (!(pseudoRandomEnabled && pseudoRandomCpuEnabled)) {
                getCPUStartingKey(tRangeStart, tRangeEnd, key, startP);
        }
        uint64_t pseudoSequentialIndex = 0;

        Int* dx = new Int[cpuGroupSize / 2 + 1];
        Point* pts = new Point[cpuGroupSize];

	Int* dy = new Int();
	Int* dyn = new Int();
	Int* _s = new Int();
	Int* _p = new Int();
	Point* pp = new Point();
	Point* pn = new Point();
	grp->Set(dx);

	ph->hasStarted = true;
	ph->rKeyRequest = false;

        while (!endOfSearch) {

                if (pseudoRandomEnabled && pseudoRandomCpuEnabled) {
                        if (!acquirePseudoRandomBlock(key, startP, pseudoSequentialIndex)) {
                                break;
                        }
                }
                else {
                        if (ph->rKeyRequest) {
                                getCPUStartingKey(tRangeStart, tRangeEnd, key, startP);
                                ph->rKeyRequest = false;
                        }
                }

		// Fill group
		int i;
                int hLength = (cpuGroupSize / 2 - 1);

		for (i = 0; i < hLength; i++) {
			dx[i].ModSub(&Gn[i].x, &startP.x);
		}
		dx[i].ModSub(&Gn[i].x, &startP.x);  // For the first point
		dx[i + 1].ModSub(&_2Gn.x, &startP.x); // For the next center point

		// Grouped ModInv
		grp->ModInv();

		// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
		// We compute key in the positive and negative way from the center of the group

		// center point
                pts[cpuGroupSize / 2] = startP;

		for (i = 0; i < hLength && !endOfSearch; i++) {

			*pp = startP;
			*pn = startP;

			// P = startP + i*G
			dy->ModSub(&Gn[i].y, &pp->y);

			_s->ModMulK1(dy, &dx[i]);       // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
			_p->ModSquareK1(_s);            // _p = pow2(s)

			pp->x.ModNeg();
			pp->x.ModAdd(_p);
			pp->x.ModSub(&Gn[i].x);           // rx = pow2(s) - p1.x - p2.x;

			pp->y.ModSub(&Gn[i].x, &pp->x);
			pp->y.ModMulK1(_s);
			pp->y.ModSub(&Gn[i].y);           // ry = - p2.y - s*(ret.x-p2.x);

			// P = startP - i*G  , if (x,y) = i*G then (x,-y) = -i*G
			dyn->Set(&Gn[i].y);
			dyn->ModNeg();
			dyn->ModSub(&pn->y);

			_s->ModMulK1(dyn, &dx[i]);      // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
			_p->ModSquareK1(_s);            // _p = pow2(s)

			pn->x.ModNeg();
			pn->x.ModAdd(_p);
			pn->x.ModSub(&Gn[i].x);          // rx = pow2(s) - p1.x - p2.x;

			pn->y.ModSub(&Gn[i].x, &pn->x);
			pn->y.ModMulK1(_s);
			pn->y.ModAdd(&Gn[i].y);          // ry = - p2.y - s*(ret.x-p2.x);

                        pts[cpuGroupSize / 2 + (i + 1)] = *pp;
                        pts[cpuGroupSize / 2 - (i + 1)] = *pn;

		}

		// First point (startP - (GRP_SZIE/2)*G)
		*pn = startP;
		dyn->Set(&Gn[i].y);
		dyn->ModNeg();
		dyn->ModSub(&pn->y);

		_s->ModMulK1(dyn, &dx[i]);
		_p->ModSquareK1(_s);

		pn->x.ModNeg();
		pn->x.ModAdd(_p);
		pn->x.ModSub(&Gn[i].x);

		pn->y.ModSub(&Gn[i].x, &pn->x);
		pn->y.ModMulK1(_s);
		pn->y.ModAdd(&Gn[i].y);

		pts[0] = *pn;

                if (!(pseudoRandomEnabled && pseudoRandomCpuEnabled)) {
                        // Next start point (startP + GRP_SIZE*G)
                        *pp = startP;
                        dy->ModSub(&_2Gn.y, &pp->y);

                        _s->ModMulK1(dy, &dx[i + 1]);
                        _p->ModSquareK1(_s);

                        pp->x.ModNeg();
                        pp->x.ModAdd(_p);
                        pp->x.ModSub(&_2Gn.x);

                        pp->y.ModSub(&_2Gn.x, &pp->x);
                        pp->y.ModMulK1(_s);
                        pp->y.ModSub(&_2Gn.y);
                        startP = *pp;
                }

                // Check addresses
		if (useSSE) {
                        for (int i = 0; i < cpuGroupSize && !endOfSearch; i += 4) {
				switch (compMode) {
				case SEARCH_COMPRESSED:
					if (searchMode == (int)SEARCH_MODE_MA) {
						checkMultiAddressesSSE(true, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					}
					else if (searchMode == (int)SEARCH_MODE_SA) {
						checkSingleAddressesSSE(true, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					}
					break;
				case SEARCH_UNCOMPRESSED:
					if (searchMode == (int)SEARCH_MODE_MA) {
						checkMultiAddressesSSE(false, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					}
					else if (searchMode == (int)SEARCH_MODE_SA) {
						checkSingleAddressesSSE(false, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					}
					break;
				case SEARCH_BOTH:
					if (searchMode == (int)SEARCH_MODE_MA) {
						checkMultiAddressesSSE(true, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
						checkMultiAddressesSSE(false, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					}
					else if (searchMode == (int)SEARCH_MODE_SA) {
						checkSingleAddressesSSE(true, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
						checkSingleAddressesSSE(false, key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
					}
					break;
				}
			}
		}
		else {
			if (coinType == COIN_BTC) {
                                for (int i = 0; i < cpuGroupSize && !endOfSearch; i++) {
					switch (compMode) {
					case SEARCH_COMPRESSED:
						switch (searchMode) {
						case (int)SEARCH_MODE_MA:
							checkMultiAddresses(true, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_SA:
							checkSingleAddress(true, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_MX:
							checkMultiXPoints(true, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_SX:
							checkSingleXPoint(true, key, i, pts[i]);
							break;
						default:
							break;
						}
						break;
					case SEARCH_UNCOMPRESSED:
						switch (searchMode) {
						case (int)SEARCH_MODE_MA:
							checkMultiAddresses(false, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_SA:
							checkSingleAddress(false, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_MX:
							checkMultiXPoints(false, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_SX:
							checkSingleXPoint(false, key, i, pts[i]);
							break;
						default:
							break;
						}
						break;
					case SEARCH_BOTH:
						switch (searchMode) {
						case (int)SEARCH_MODE_MA:
							checkMultiAddresses(true, key, i, pts[i]);
							checkMultiAddresses(false, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_SA:
							checkSingleAddress(true, key, i, pts[i]);
							checkSingleAddress(false, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_MX:
							checkMultiXPoints(true, key, i, pts[i]);
							checkMultiXPoints(false, key, i, pts[i]);
							break;
						case (int)SEARCH_MODE_SX:
							checkSingleXPoint(true, key, i, pts[i]);
							checkSingleXPoint(false, key, i, pts[i]);
							break;
						default:
							break;
						}
						break;
					}
				}
			}
			else {
                                for (int i = 0; i < cpuGroupSize && !endOfSearch; i++) {
					switch (searchMode) {
					case (int)SEARCH_MODE_MA:
						checkMultiAddressesETH(key, i, pts[i]);
						break;
					case (int)SEARCH_MODE_SA:
						checkSingleAddressETH(key, i, pts[i]);
						break;
					default:
						break;
					}
				}
			}
                }
                if (!(pseudoRandomEnabled && pseudoRandomCpuEnabled)) {
                        key.Add(static_cast<uint64_t>(cpuGroupSize));
                }
                else {
                        notifyPseudoRandomBlockComplete(pseudoSequentialIndex);
                }
                counters[thId] += cpuGroupSize; // Point
        }
	ph->isRunning = false;

	delete grp;
	delete[] dx;
	delete[] pts;

	delete dy;
	delete dyn;
	delete _s;
	delete _p;
	delete pp;
	delete pn;
}

// ----------------------------------------------------------------------------

void KeyHunt::getGPUStartingKeys(Int & tRangeStart, Int & tRangeEnd, int compiledGroupSize, int nbThread, Int * keys, Point * p)
{

        Int tRangeDiff(tRangeEnd);
        Int tRangeStart2(tRangeStart);
        Int tRangeEnd2(tRangeStart);

	Int tThreads;
	tThreads.SetInt32(nbThread);
	tRangeDiff.Set(&tRangeEnd);
	tRangeDiff.Sub(&tRangeStart);
	tRangeDiff.Div(&tThreads);

	int rangeShowThreasold = 3;
	int rangeShowCounter = 0;

        const uint64_t midpoint = static_cast<uint64_t>(compiledGroupSize) / 2ULL;

	for (int i = 0; i < nbThread; i++) {

                tRangeEnd2.Set(&tRangeStart2);
                tRangeEnd2.Add(&tRangeDiff);

		if (rKey <= 0)
			keys[i].Set(&tRangeStart2);
		else
			keys[i].Rand(&tRangeEnd2);

		tRangeStart2.Add(&tRangeDiff);

		Int k(keys + i);
		k.Add(midpoint);	// Starting key is at the middle of the GPU group
		p[i] = secp->ComputePublicKey(&k);
	}

}

void KeyHunt::FindKeyGPU(TH_PARAM * ph)
{

	bool ok = true;

#ifdef WITHGPU

	// Global init
	int thId = ph->threadId;
	Int tRangeStart = ph->rangeStart;
	Int tRangeEnd = ph->rangeEnd;

        const bool retainInputKeys = (rKey != 0) || pseudoRandomEnabled;
        GPUEngine* g;
        switch (searchMode) {
        case (int)SEARCH_MODE_MA:
        case (int)SEARCH_MODE_MX:
                g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
                        BLOOM_N, bloom->get_bits(), bloom->get_hashes(), bloom->get_bf(), DATA, TOTAL_COUNT, retainInputKeys,
                        gpuStepMultiplierEffective);
                break;
        case (int)SEARCH_MODE_SA:
                g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
                        hash160Keccak, retainInputKeys, gpuStepMultiplierEffective);
                break;
        case (int)SEARCH_MODE_SX:
                g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
                        xpoint, retainInputKeys, gpuStepMultiplierEffective);
                break;
        default:
                printf("Invalid search mode format");
                return;
                break;
        }


        int nbThread = g->GetNbThread();
        int threadsPerGroup = g->GetThreadsPerGroup();
        Point* pointBuffers[2] = { new Point[nbThread], new Point[nbThread] };
        Int* keyBuffers[2] = { new Int[nbThread], new Int[nbThread] };
        std::vector<uint64_t> pseudoSequentialBuffers[2] = {
                std::vector<uint64_t>(nbThread, std::numeric_limits<uint64_t>::max()),
                std::vector<uint64_t>(nbThread, std::numeric_limits<uint64_t>::max()) };
        int currentBuffer = 0;
        int nextBuffer = 1;
        int currentAssignedBlocks = 0;

        const int compiledGroupSize = GPUEngine::GetCompiledGroupSize();
        const uint64_t compiledGroupMidpoint = static_cast<uint64_t>(compiledGroupSize) / 2ULL;

        auto preparePseudoRandomBatch = [&](int bufferIndex, int& assignedBlocksOut, int& activeThreadsOut) {
                auto& sequential = pseudoSequentialBuffers[bufferIndex];
                std::fill(sequential.begin(), sequential.end(), std::numeric_limits<uint64_t>::max());

                assignedBlocksOut = 0;
                activeThreadsOut = 0;

                for (int i = 0; i < nbThread; i++) {
                        PseudoRandomBlock block;
                        if (!dequeuePseudoRandomGpuBlock(block)) {
                                break;
                        }

                        keyBuffers[bufferIndex][i] = block.key;
                        Int startScalar(keyBuffers[bufferIndex] + i);
                        startScalar.Add(compiledGroupMidpoint);
                        pointBuffers[bufferIndex][i] = secp->ComputePublicKey(&startScalar);
                        sequential[i] = block.sequentialIndex;
                        assignedBlocksOut++;
                }

                if (assignedBlocksOut == 0) {
                        return false;
                }

                activeThreadsOut = assignedBlocksOut;
                if (threadsPerGroup > 0) {
                        int remainder = activeThreadsOut % threadsPerGroup;
                        if (remainder != 0) {
                                int pad = threadsPerGroup - remainder;
                                if (pad > nbThread - activeThreadsOut) {
                                        pad = nbThread - activeThreadsOut;
                                }
                                for (int j = 0; j < pad; j++) {
                                        int idx = activeThreadsOut + j;
                                        if (idx >= nbThread) {
                                                break;
                                        }
                                        pointBuffers[bufferIndex][idx] = pointBuffers[bufferIndex][assignedBlocksOut - 1];
                                        keyBuffers[bufferIndex][idx] = keyBuffers[bufferIndex][assignedBlocksOut - 1];
                                        sequential[idx] = std::numeric_limits<uint64_t>::max();
                                }
                                activeThreadsOut += pad;
                        }
                }

                return true;
        };
        std::vector<ITEM> found;
        printf("GPU          : %s\n\n", g->deviceName.c_str());

        counters[thId] = 0;

        const bool usePseudoRandomGpu = pseudoRandomEnabled;
        if (usePseudoRandomGpu) {
                startPseudoRandomGpuPrefetch(nbThread);
        }
        if (usePseudoRandomGpu) {
                int initialActiveThreads = 0;
                bool haveInitial = preparePseudoRandomBatch(currentBuffer, currentAssignedBlocks, initialActiveThreads);
                if (haveInitial) {
                        ok = g->SetKeys(pointBuffers[currentBuffer], initialActiveThreads);
                }
                else {
                        ok = false;
                }
        }
        else {
                getGPUStartingKeys(tRangeStart, tRangeEnd, compiledGroupSize, nbThread, keyBuffers[currentBuffer],
                        pointBuffers[currentBuffer]);
                ok = g->SetKeys(pointBuffers[currentBuffer]);
        }

        ph->hasStarted = true;
        ph->rKeyRequest = false;

        // GPU Thread
        while (ok && !endOfSearch) {

                const int resultsBuffer = currentBuffer;
                Int* resultKeys = keyBuffers[resultsBuffer];
                auto& resultSequential = pseudoSequentialBuffers[resultsBuffer];
                const int resultAssignedBlocks = currentAssignedBlocks;

                int assignedBlocks = 0;
                int activeThreads = nbThread;

                if (usePseudoRandomGpu && ph->rKeyRequest) {
                        ph->rKeyRequest = false;
                }

                if (usePseudoRandomGpu) {
                        preparePseudoRandomBatch(nextBuffer, assignedBlocks, activeThreads);
                }
                else {
                        if (ph->rKeyRequest) {
                                getGPUStartingKeys(tRangeStart, tRangeEnd, compiledGroupSize, nbThread, keyBuffers[currentBuffer],
                                        pointBuffers[currentBuffer]);
                                ok = g->SetKeys(pointBuffers[currentBuffer]);
                                ph->rKeyRequest = false;
                        }
                }

                const bool queueNextBatch = !usePseudoRandomGpu;

                // Call kernel
                switch (searchMode) {
                case (int)SEARCH_MODE_MA:
                        ok = g->LaunchSEARCH_MODE_MA(found, false, queueNextBatch);
                        for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
                                ITEM it = found[i];
                                if (coinType == COIN_BTC) {
                                        std::string addr = secp->GetAddress(it.mode, it.hash);
                                        if (checkPrivKey(addr, resultKeys[it.thId], it.incr, it.mode)) {
                                                nbFoundKey++;
                                        }
                                }
                                else {
                                        std::string addr = secp->GetAddressETH(it.hash);
                                        if (checkPrivKeyETH(addr, resultKeys[it.thId], it.incr)) {
                                                nbFoundKey++;
                                        }
                                }
                        }
                        break;
                case (int)SEARCH_MODE_MX:
                        ok = g->LaunchSEARCH_MODE_MX(found, false, queueNextBatch);
                        for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
                                ITEM it = found[i];
                                if (checkPrivKeyX(/*addr,*/ resultKeys[it.thId], it.incr, it.mode)) {
                                        nbFoundKey++;
                                }
                        }
                        break;
                case (int)SEARCH_MODE_SA:
                        ok = g->LaunchSEARCH_MODE_SA(found, false, queueNextBatch);
                        for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
                                ITEM it = found[i];
                                if (coinType == COIN_BTC) {
                                        std::string addr = secp->GetAddress(it.mode, it.hash);
                                        if (checkPrivKey(addr, resultKeys[it.thId], it.incr, it.mode)) {
                                                nbFoundKey++;
                                        }
                                }
                                else {
                                        std::string addr = secp->GetAddressETH(it.hash);
                                        if (checkPrivKeyETH(addr, resultKeys[it.thId], it.incr)) {
                                                nbFoundKey++;
                                        }
                                }
                        }
                        break;
                case (int)SEARCH_MODE_SX:
                        ok = g->LaunchSEARCH_MODE_SX(found, false, queueNextBatch);
                        for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
                                ITEM it = found[i];
                                if (checkPrivKeyX(/*addr,*/ resultKeys[it.thId], it.incr, it.mode)) {
                                        nbFoundKey++;
                                }
                        }
                        break;
                default:
                        break;
                }

                if (ok) {
                        const uint64_t stepSize = g->GetStepSize();
                        if (usePseudoRandomGpu) {
                                for (int i = 0; i < resultAssignedBlocks; i++) {
                                        if (resultSequential[i] != std::numeric_limits<uint64_t>::max()) {
                                                notifyPseudoRandomBlockComplete(resultSequential[i]);
                                        }
                                }
                                counters[thId] += stepSize * static_cast<uint64_t>(resultAssignedBlocks);
                        }
                        else {
                                for (int i = 0; i < nbThread; i++) {
                                        resultKeys[i].Add(stepSize);
                                }
                                counters[thId] += stepSize * static_cast<uint64_t>(nbThread); // Point
                        }
                }

                if (!ok || endOfSearch) {
                        break;
                }

                if (!usePseudoRandomGpu) {
                        continue;
                }

                if (assignedBlocks == 0) {
                        ok = false;
                        break;
                }

                ok = g->SetKeys(pointBuffers[nextBuffer], activeThreads);
                if (!ok) {
                        break;
                }

                currentBuffer = nextBuffer;
                currentAssignedBlocks = assignedBlocks;
                nextBuffer = 1 - currentBuffer;

                }

        if (usePseudoRandomGpu) {
                stopPseudoRandomGpuPrefetch();
        }

        delete[] keyBuffers[0];
        delete[] keyBuffers[1];
        delete[] pointBuffers[0];
        delete[] pointBuffers[1];
        delete g;

#else
	ph->hasStarted = true;
	printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

	ph->isRunning = false;

}

// ----------------------------------------------------------------------------

bool KeyHunt::isAlive(TH_PARAM * p)
{

	bool isAlive = true;
	int total = nbCPUThread + nbGPUThread;
	for (int i = 0; i < total; i++)
		isAlive = isAlive && p[i].isRunning;

	return isAlive;

}

// ----------------------------------------------------------------------------

bool KeyHunt::hasStarted(TH_PARAM * p)
{

	bool hasStarted = true;
	int total = nbCPUThread + nbGPUThread;
	for (int i = 0; i < total; i++)
		hasStarted = hasStarted && p[i].hasStarted;

	return hasStarted;

}

// ----------------------------------------------------------------------------

uint64_t KeyHunt::getGPUCount()
{

	uint64_t count = 0;
	for (int i = 0; i < nbGPUThread; i++)
		count += counters[0x80L + i];
	return count;

}

// ----------------------------------------------------------------------------

uint64_t KeyHunt::getCPUCount()
{

	uint64_t count = 0;
	for (int i = 0; i < nbCPUThread; i++)
		count += counters[i];
	return count;

}

// ----------------------------------------------------------------------------

void KeyHunt::rKeyRequest(TH_PARAM * p) {

	int total = nbCPUThread + nbGPUThread;
	for (int i = 0; i < total; i++)
		p[i].rKeyRequest = true;

}
// ----------------------------------------------------------------------------

void KeyHunt::SetupRanges(uint32_t totalThreads)
{
	Int threads;
	threads.SetInt32(totalThreads);
	rangeDiff.Set(&rangeEnd);
	rangeDiff.Sub(&rangeStart);
	rangeDiff.Div(&threads);
}

// ----------------------------------------------------------------------------

void KeyHunt::Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit)
{

	double t0;
	double t1;
	endOfSearch = false;
	nbCPUThread = nbThread;
	nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
	nbFoundKey = 0;

	// setup ranges
	SetupRanges(nbCPUThread + nbGPUThread);

	memset(counters, 0, sizeof(counters));

	if (!useGpu)
		printf("\n");

	TH_PARAM* params = (TH_PARAM*)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
	memset(params, 0, (nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));

	// Launch CPU threads
	for (int i = 0; i < nbCPUThread; i++) {
		params[i].obj = this;
		params[i].threadId = i;
		params[i].isRunning = true;

		params[i].rangeStart.Set(&rangeStart);
		rangeStart.Add(&rangeDiff);
		params[i].rangeEnd.Set(&rangeStart);

#ifdef WIN64
		DWORD thread_id;
		CreateThread(NULL, 0, _FindKeyCPU, (void*)(params + i), 0, &thread_id);
		ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
		pthread_t thread_id;
		pthread_create(&thread_id, NULL, &_FindKeyCPU, (void*)(params + i));
		ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
	}

	// Launch GPU threads
	for (int i = 0; i < nbGPUThread; i++) {
		params[nbCPUThread + i].obj = this;
		params[nbCPUThread + i].threadId = 0x80L + i;
		params[nbCPUThread + i].isRunning = true;
		params[nbCPUThread + i].gpuId = gpuId[i];
		params[nbCPUThread + i].gridSizeX = gridSize[2 * i];
		params[nbCPUThread + i].gridSizeY = gridSize[2 * i + 1];

		params[nbCPUThread + i].rangeStart.Set(&rangeStart);
		rangeStart.Add(&rangeDiff);
		params[nbCPUThread + i].rangeEnd.Set(&rangeStart);


#ifdef WIN64
		DWORD thread_id;
		CreateThread(NULL, 0, _FindKeyGPU, (void*)(params + (nbCPUThread + i)), 0, &thread_id);
#else
		pthread_t thread_id;
		pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(params + (nbCPUThread + i)));
#endif
	}

#ifndef WIN64
	setvbuf(stdout, NULL, _IONBF, 0);
#endif
	printf("\n");

	uint64_t lastCount = 0;
	uint64_t gpuCount = 0;
	uint64_t lastGPUCount = 0;

	// Key rate smoothing filter
#define FILTER_SIZE 8
	double lastkeyRate[FILTER_SIZE];
	double lastGpukeyRate[FILTER_SIZE];
	uint32_t filterPos = 0;

	double keyRate = 0.0;
	double gpuKeyRate = 0.0;
	char timeStr[256];

	memset(lastkeyRate, 0, sizeof(lastkeyRate));
	memset(lastGpukeyRate, 0, sizeof(lastkeyRate));

	// Wait that all threads have started
	while (!hasStarted(params)) {
		Timer::SleepMillis(500);
	}

	// Reset timer
	Timer::Init();
	t0 = Timer::get_tick();
	startTime = t0;
	Int p100;
	Int ICount;
	p100.SetInt32(100);
	double completedPerc = 0;
	uint64_t rKeyCount = 0;
	while (isAlive(params)) {

		int delay = 2000;
		while (isAlive(params) && delay > 0) {
			Timer::SleepMillis(500);
			delay -= 500;
		}

		gpuCount = getGPUCount();
		uint64_t count = getCPUCount() + gpuCount;
		ICount.SetInt64(count);
		int completedBits = ICount.GetBitLength();
		if (rKey <= 0) {
			completedPerc = CalcPercantage(ICount, rangeStart, rangeDiff2);
			//ICount.Mult(&p100);
			//ICount.Div(&this->rangeDiff2);
			//completedPerc = std::stoi(ICount.GetBase10());
		}

		t1 = Timer::get_tick();
		keyRate = (double)(count - lastCount) / (t1 - t0);
		gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
		lastkeyRate[filterPos % FILTER_SIZE] = keyRate;
		lastGpukeyRate[filterPos % FILTER_SIZE] = gpuKeyRate;
		filterPos++;

		// KeyRate smoothing
		double avgKeyRate = 0.0;
		double avgGpuKeyRate = 0.0;
		uint32_t nbSample;
		for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
			avgKeyRate += lastkeyRate[nbSample];
			avgGpuKeyRate += lastGpukeyRate[nbSample];
		}
		avgKeyRate /= (double)(nbSample);
		avgGpuKeyRate /= (double)(nbSample);

		if (isAlive(params)) {
			memset(timeStr, '\0', 256);
			printf("\r[%s] [CPU+GPU: %.2f Mk/s] [GPU: %.2f Mk/s] [C: %lf %%] [R: %" PRIu64 "] [T: %s (%d bit)] [F: %d]  ",
				toTimeStr(t1, timeStr),
				avgKeyRate / 1000000.0,
				avgGpuKeyRate / 1000000.0,
				completedPerc,
				rKeyCount,
				formatThousands(count).c_str(),
				completedBits,
				nbFoundKey);
		}
		if (rKey > 0) {
			if ((count - lastrKey) > (1000000 * rKey)) {
				// rKey request
				rKeyRequest(params);
				lastrKey = count;
				rKeyCount++;
			}
		}

		lastCount = count;
		lastGPUCount = gpuCount;
		t0 = t1;
		if (should_exit || nbFoundKey >= targetCounter || completedPerc > 100.5)
			endOfSearch = true;
	}

	free(params);

	}

// ----------------------------------------------------------------------------

std::string KeyHunt::GetHex(std::vector<unsigned char> &buffer)
{
	std::string ret;

	char tmp[128];
	for (int i = 0; i < (int)buffer.size(); i++) {
		sprintf(tmp, "%02X", buffer[i]);
		ret.append(tmp);
	}
	return ret;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int KeyHunt::CheckBloomBinary(const uint8_t * _xx, uint32_t K_LENGTH)
{
	if (bloom->check(_xx, K_LENGTH) > 0) {
		uint8_t* temp_read;
		uint64_t half, min, max, current; //, current_offset
		int64_t rcmp;
		int32_t r = 0;
		min = 0;
		current = 0;
		max = TOTAL_COUNT;
		half = TOTAL_COUNT;
		while (!r && half >= 1) {
			half = (max - min) / 2;
			temp_read = DATA + ((current + half) * K_LENGTH);
			rcmp = memcmp(_xx, temp_read, K_LENGTH);
			if (rcmp == 0) {
				r = 1;  //Found!!
			}
			else {
				if (rcmp < 0) { //data < temp_read
					max = (max - half);
				}
				else { // data > temp_read
					min = (min + half);
				}
				current = min;
			}
		}
		return r;
	}
	return 0;
}

// ----------------------------------------------------------------------------

bool KeyHunt::MatchHash(uint32_t * _h)
{
	if (_h[0] == hash160Keccak[0] &&
		_h[1] == hash160Keccak[1] &&
		_h[2] == hash160Keccak[2] &&
		_h[3] == hash160Keccak[3] &&
		_h[4] == hash160Keccak[4]) {
		return true;
	}
	else {
		return false;
	}
}

// ----------------------------------------------------------------------------

bool KeyHunt::MatchXPoint(uint32_t * _h)
{
	if (_h[0] == xpoint[0] &&
		_h[1] == xpoint[1] &&
		_h[2] == xpoint[2] &&
		_h[3] == xpoint[3] &&
		_h[4] == xpoint[4] &&
		_h[5] == xpoint[5] &&
		_h[6] == xpoint[6] &&
		_h[7] == xpoint[7]) {
		return true;
	}
	else {
		return false;
	}
}

// ----------------------------------------------------------------------------

std::string KeyHunt::formatThousands(uint64_t x)
{
	char buf[32] = "";

	snprintf(buf, sizeof(buf), "%" PRIu64, x);

	std::string s(buf);

	int len = (int)s.length();

	int numCommas = (len - 1) / 3;

	if (numCommas == 0) {
		return s;
	}

	std::string result = "";

	int count = ((len % 3) == 0) ? 0 : (3 - (len % 3));

	for (int i = 0; i < len; i++) {
		result += s[i];

		if (count++ == 2 && i < len - 1) {
			result += ",";
			count = 0;
		}
	}
	return result;
}

// ----------------------------------------------------------------------------

char* KeyHunt::toTimeStr(int sec, char* timeStr)
{
	int h, m, s;
	h = (sec / 3600);
	m = (sec - (3600 * h)) / 60;
	s = (sec - (3600 * h) - (m * 60));
	sprintf(timeStr, "%0*d:%0*d:%0*d", 2, h, 2, m, 2, s);
	return (char*)timeStr;
}

// ----------------------------------------------------------------------------

//#include <gmp.h>
//#include <gmpxx.h>
// ((input - min) * 100) / (max - min)
//double KeyHunt::GetPercantage(uint64_t v)
//{
//	//Int val(v);
//	//mpz_class x(val.GetBase16().c_str(), 16);
//	//mpz_class r(rangeStart.GetBase16().c_str(), 16);
//	//x = x - mpz_class(rangeEnd.GetBase16().c_str(), 16);
//	//x = x * 100;
//	//mpf_class y(x);
//	//y = y / mpf_class(r);
//	return 0;// y.get_d();
//}




