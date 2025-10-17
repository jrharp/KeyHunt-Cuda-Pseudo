#ifndef KEYHUNTH
#define KEYHUNTH

#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include "SECP256k1.h"
#include "Bloom.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#endif

static constexpr int CPU_GRP_SIZE = 1024 * 2;

class KeyHunt;

typedef struct {
	KeyHunt* obj;
	int  threadId;
	bool isRunning;
	bool hasStarted;

	int  gridSizeX;
	int  gridSizeY;
	int  gpuId;

	Int rangeStart;
	Int rangeEnd;
	bool rKeyRequest;
} TH_PARAM;


class KeyHunt
{

public:

	KeyHunt(const std::string& inputFile, int compMode, int searchMode, int coinType, bool useGpu, 
		const std::string& outputFile, bool useSSE, uint32_t maxFound, uint64_t rKey, 
		const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit);

	KeyHunt(const std::vector<unsigned char>& hashORxpoint, int compMode, int searchMode, int coinType, 
		bool useGpu, const std::string& outputFile, bool useSSE, uint32_t maxFound, uint64_t rKey, 
		const std::string& rangeStart, const std::string& rangeEnd, bool& should_exit);

	~KeyHunt();

	void Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit);
	void FindKeyCPU(TH_PARAM* p);
	void FindKeyGPU(TH_PARAM* p);

private:

        void InitGenratorTable();
        void initializePseudoRandomState();
        bool acquirePseudoRandomBlock(Int& key, Point& startP, uint64_t& sequentialIndex);
        uint64_t permuteBlockIndex(uint64_t value) const;
        void persistPseudoRandomState(uint64_t completedCount);
        bool loadPseudoRandomState(uint64_t& resumeIndex) const;
        void notifyPseudoRandomBlockComplete(uint64_t sequentialIndex);
        struct PseudoRandomBlock;
        void startPseudoRandomGpuPrefetch(int targetQueueSize);
        void stopPseudoRandomGpuPrefetch();
        bool dequeuePseudoRandomGpuBlock(PseudoRandomBlock& block);
        void pseudoRandomGpuWorker();

        std::string GetHex(std::vector<unsigned char>& buffer);
        bool checkPrivKey(std::string addr, Int& key, int32_t incr, bool mode);
        bool checkPrivKeyETH(std::string addr, Int& key, int32_t incr);
        bool checkPrivKeyX(Int& key, int32_t incr, bool mode);

	void checkMultiAddresses(bool compressed, Int key, int i, Point p1);
	void checkMultiAddressesETH(Int key, int i, Point p1);
	void checkSingleAddress(bool compressed, Int key, int i, Point p1);
	void checkSingleAddressETH(Int key, int i, Point p1);
	void checkMultiXPoints(bool compressed, Int key, int i, Point p1);
	void checkSingleXPoint(bool compressed, Int key, int i, Point p1);

	void checkMultiAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);
	void checkSingleAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);

	void output(std::string addr, std::string pAddr, std::string pAddrHex, std::string pubKey);
	bool isAlive(TH_PARAM* p);

	bool hasStarted(TH_PARAM* p);
	uint64_t getGPUCount();
	uint64_t getCPUCount();
	void rKeyRequest(TH_PARAM* p);
	void SetupRanges(uint32_t totalThreads);

	void getCPUStartingKey(Int& tRangeStart, Int& tRangeEnd, Int& key, Point& startP);
	void getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Int* keys, Point* p);

	int CheckBloomBinary(const uint8_t* _xx, uint32_t K_LENGTH);
	bool MatchHash(uint32_t* _h);
	bool MatchXPoint(uint32_t* _h);
	std::string formatThousands(uint64_t x);
	char* toTimeStr(int sec, char* timeStr);

	Secp256K1* secp;
	Bloom* bloom;

	uint64_t counters[256];
	double startTime;

	int compMode;
	int searchMode;
	int coinType;

	bool useGpu;
	bool endOfSearch;
	int nbCPUThread;
	int nbGPUThread;
	int nbFoundKey;
	uint64_t targetCounter;

	std::string outputFile;
	std::string inputFile;
	uint32_t hash160Keccak[5];
	uint32_t xpoint[8];
        bool useSSE;

        int cpuGroupSize = CPU_GRP_SIZE;

	Int rangeStart;
	Int rangeEnd;
	Int rangeDiff;
	Int rangeDiff2;

	uint32_t maxFound;
	uint64_t rKey;
	uint64_t lastrKey;

	uint8_t* DATA;
	uint64_t TOTAL_COUNT;
        uint64_t BLOOM_N;

        struct PseudoRandomState {
                uint64_t totalKeys = 0;
                uint64_t totalBlocks = 0;
                uint64_t blockMask = 0;
                unsigned int blockBits = 0;
                uint64_t blockKeyCount = 0;
                std::atomic<uint64_t> nextCounter{ 0 };
                std::string stateFile;
                mutable std::mutex fileMutex;
                mutable std::mutex progressMutex;
                std::unordered_set<uint64_t> completedBlocks;
                uint64_t lowestUnpersisted = 0;
                uint64_t lastPersisted = std::numeric_limits<uint64_t>::max();
                bool persistWarningShown = false;
        };

        bool pseudoRandomEnabled = false;
        bool pseudoRandomCpuEnabled = false;
        PseudoRandomState pseudoState;
        Int initialRangeStart;

        struct PseudoRandomBlock {
                Int key;
                Point startPoint;
                uint64_t sequentialIndex = 0;
        };

        std::vector<std::thread> pseudoGpuWorkers;
        std::deque<PseudoRandomBlock> pseudoGpuQueue;
        std::mutex pseudoGpuMutex;
        std::condition_variable pseudoGpuCv;
        std::atomic<int> pseudoGpuActiveWorkers{ 0 };
        size_t pseudoGpuQueueLimit = 0;
        std::atomic<bool> pseudoGpuStop{ false };

#ifdef WIN64
        HANDLE ghMutex;
#else
        pthread_mutex_t  ghMutex;
#endif

};

#endif // KEYHUNTH
