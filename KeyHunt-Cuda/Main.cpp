#include "Timer.h"
#include "KeyHunt.h"
#include "Base58.h"
#include "CmdParse.h"
#include "EmailNotifier.h"
#ifdef WITHGPU
#include "GPU/GPUEngine.h"
#endif
#include <fstream>
#include <string>
#include <sstream>
#include <string.h>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <limits>
#include <inttypes.h>
#ifndef WIN64
#include <signal.h>
#include <unistd.h>
#endif

#define RELEASE "1.07"

using namespace std;
bool should_exit = false;
#ifdef WIN64
static bool externalStopRequested = false;
#else
static volatile sig_atomic_t externalStopRequested = 0;
#endif

namespace {

std::string buildShutdownSummary()
{
#ifdef WIN64
        const bool interrupted = externalStopRequested;
#else
        const bool interrupted = externalStopRequested != 0;
#endif

        if (interrupted) {
                return "KeyHunt stopped due to an external interruption.";
        }

        if (should_exit) {
                return "KeyHunt stopped after reaching the configured termination condition.";
        }

        return "KeyHunt finished execution normally.";
}

std::string describeSearchMode(int searchMode)
{
        switch (searchMode) {
        case (int)SEARCH_MODE_MA:
                return "Multi Address";
        case (int)SEARCH_MODE_SA:
                return "Single Address";
        case (int)SEARCH_MODE_MX:
                return "Multi X Points";
        case (int)SEARCH_MODE_SX:
                return "Single X Point";
        default:
                return "Unknown";
        }
}

std::string describeCompressionMode(int compMode)
{
        switch (compMode) {
        case SEARCH_COMPRESSED:
                return "COMPRESSED";
        case SEARCH_UNCOMPRESSED:
                return "UNCOMPRESSED";
        case SEARCH_BOTH:
                return "COMPRESSED & UNCOMPRESSED";
        default:
                return "Unknown";
        }
}

std::string joinIntegerList(const std::vector<int>& values)
{
        if (values.empty()) {
                return "(none)";
        }

        std::ostringstream oss;
        for (size_t i = 0; i < values.size(); ++i) {
                if (i > 0) {
                        oss << ", ";
                }
                oss << values[i];
        }
        return oss.str();
}

std::string formatGpuGridSize(const std::vector<int>& gridSize)
{
        if (gridSize.empty()) {
                return "(default)";
        }

        std::ostringstream oss;
        for (size_t i = 0; i < gridSize.size(); ++i) {
                oss << gridSize[i];
                if (i + 1 < gridSize.size()) {
                        if (((i + 1) % 2) != 0) {
                                oss << "x";
                        }
                        else {
                                oss << ", ";
                        }
                }
        }
        return oss.str();
}

std::string buildStartupSummary(const std::string& release,
                int coinType,
                int compMode,
                int searchMode,
                const char* deviceLabel,
                bool useGpuDevice,
                bool useCpuDevice,
                int nbCPUThread,
                const std::vector<int>& gpuId,
                const std::vector<int>& gridSize,
                bool gpuAutoGrid,
                bool gpuUseOccupancyBlockSize,
                bool useSSE,
                uint64_t rKey,
                uint32_t maxFound,
                const std::string& inputFile,
                const std::string& address,
                const std::string& xpoint,
                const std::string& outputFile,
                const std::string& rangeStartHex,
                const std::string& rangeEndHex,
                const std::string& shardLabel)
{
        std::ostringstream summary;
        summary << "KeyHunt-Cuda v" << release << " has started." << "\n\n";
        summary << "Coin type     : " << (coinType == COIN_BTC ? "BITCOIN" : "ETHEREUM") << "\n";
        summary << "Search mode   : " << describeSearchMode(searchMode) << "\n";
        if (coinType == COIN_BTC) {
                summary << "Compression   : " << describeCompressionMode(compMode) << "\n";
        }
        summary << "Device usage  : " << deviceLabel << "\n";
        if (useCpuDevice) {
                summary << "CPU threads   : " << nbCPUThread << "\n";
        }
        if (useGpuDevice) {
                summary << "GPU ids       : " << joinIntegerList(gpuId) << "\n";
                summary << "GPU gridsize  : " << formatGpuGridSize(gridSize) << "\n";
                summary << "GPU grid mode : " << (gpuAutoGrid ? "auto-selected" : "manual");
                if (gpuUseOccupancyBlockSize) {
                        summary << " (occupancy-guided)";
                }
                summary << "\n";
        }
        summary << "SSE enabled   : " << (useSSE ? "YES" : "NO") << "\n";
        summary << "Random key    : ";
        if (rKey == 0) {
                summary << "disabled" << "\n";
        }
        else {
                summary << rKey << " Mkeys" << "\n";
        }
        summary << "Max results   : " << maxFound << "\n";
        summary << "Output file   : " << outputFile << "\n";
        switch (searchMode) {
        case (int)SEARCH_MODE_MA:
                summary << "Targets file  : " << inputFile << "\n";
                break;
        case (int)SEARCH_MODE_SA:
                summary << "Single target : "
                        << (coinType == COIN_ETH ? ("0x" + address) : address) << "\n";
                break;
        case (int)SEARCH_MODE_MX:
                summary << "XPoints file  : " << inputFile << "\n";
                break;
        case (int)SEARCH_MODE_SX:
                summary << "Single xpoint : " << xpoint << "\n";
                break;
        default:
                break;
        }
        summary << "Range start   : " << rangeStartHex << "\n";
        summary << "Range end     : " << rangeEndHex << "\n";
        summary << "Shard         : " << (shardLabel.empty() ? "(not partitioned)" : shardLabel) << "\n";
        return summary.str();
}

} // namespace

// ----------------------------------------------------------------------------
void usage()
{
	printf("KeyHunt-Cuda [OPTIONS...] [TARGETS]\n");
	printf("Where TARGETS is one address/xpont, or multiple hashes/xpoints file\n\n");

	printf("-h, --help                               : Display this message\n");
	printf("-c, --check                              : Check the working of the codes\n");
	printf("-u, --uncomp                             : Search uncompressed points\n");
	printf("-b, --both                               : Search both uncompressed or compressed points\n");
	printf("-g, --gpu                                : Enable GPU calculation (default when available)\n");
        printf("--gpui GPU ids: 0,1,...                  : List of GPU(s) to use, default is 0\n");
        printf("--gpux GPU gridsize: g0x,g0y,g1x,g1y,... : Specify GPU(s) kernel gridsize, default is 8*(Device MP count),128\n");
#ifdef WITHGPU
        printf("--gpu-autoblock                          : Use CUDA occupancy guidance to auto-select threads per block\n");
#endif
	printf("-t, --thread N                           : Specify number of CPU thread and disable GPU acceleration\n");
	printf("-i, --in FILE                            : Read rmd160 hashes or xpoints from FILE, should be in binary format with sorted\n");
	printf("-o, --out FILE                           : Write keys to FILE, default: Found.txt\n");
	printf("-m, --mode MODE                          : Specify search mode where MODE is\n");
	printf("                                               ADDRESS  : for single address\n");
	printf("                                               ADDRESSES: for multiple hashes/addresses\n");
	printf("                                               XPOINT   : for single xpoint\n");
	printf("                                               XPOINTS  : for multiple xpoints\n");
	printf("--coin BTC/ETH                           : Specify Coin name to search\n");
	printf("                                               BTC: available mode :-\n");
	printf("                                                   ADDRESS, ADDRESSES, XPOINT, XPOINTS\n");
	printf("                                               ETH: available mode :-\n");
	printf("                                                   ADDRESS, ADDRESSES\n");
	printf("-l, --list                               : List cuda enabled devices\n");
        printf("--range KEYSPACE                         : Specify the range:\n");
        printf("                                               START:END\n");
        printf("                                               START:+COUNT\n");
        printf("                                               START\n");
        printf("                                               :END\n");
        printf("                                               :+COUNT\n");
        printf("                                               Where START, END, COUNT are in hex format\n");
        printf("--shard-total COUNT                      : Split the keyspace into COUNT contiguous shards\n");
        printf("--shard-index INDEX                      : Process shard INDEX (0-based) from the split\n");
        printf("-r, --rkey Rkey                          : Random key interval in MegaKeys, default is disabled\n");
        printf("-v, --version                            : Show version\n");
}

// ----------------------------------------------------------------------------

void getInts(string name, vector<int>& tokens, const string& text, char sep)
{

	size_t start = 0, end = 0;
	tokens.clear();
	int item;

	try {

		while ((end = text.find(sep, start)) != string::npos) {
			item = std::stoi(text.substr(start, end - start));
			tokens.push_back(item);
			start = end + 1;
		}

		item = std::stoi(text.substr(start));
		tokens.push_back(item);

	}
	catch (std::invalid_argument&) {

		printf("Invalid %s argument, number expected\n", name.c_str());
		usage();
		exit(-1);

	}

}

// ----------------------------------------------------------------------------

int parseSearchMode(const std::string& s)
{
	std::string stype = s;
	std::transform(stype.begin(), stype.end(), stype.begin(), ::tolower);

	if (stype == "address") {
		return SEARCH_MODE_SA;
	}

	if (stype == "xpoint") {
		return SEARCH_MODE_SX;
	}

	if (stype == "addresses") {
		return SEARCH_MODE_MA;
	}

	if (stype == "xpoints") {
		return SEARCH_MODE_MX;
	}

	printf("Invalid search mode format: %s", stype.c_str());
	usage();
	exit(-1);
}

// ----------------------------------------------------------------------------

int parseCoinType(const std::string& s)
{
	std::string stype = s;
	std::transform(stype.begin(), stype.end(), stype.begin(), ::tolower);

	if (stype == "btc") {
		return COIN_BTC;
	}

	if (stype == "eth") {
		return COIN_ETH;
	}

	printf("Invalid coin name: %s", stype.c_str());
	usage();
	exit(-1);
}

// ----------------------------------------------------------------------------

bool parseRange(const std::string& s, Int& start, Int& end)
{
	size_t pos = s.find(':');

	if (pos == std::string::npos) {
		start.SetBase16(s.c_str());
		end.Set(&start);
		end.Add(0xFFFFFFFFFFFFULL);
	}
	else {
		std::string left = s.substr(0, pos);

		if (left.length() == 0) {
			start.SetInt32(1);
		}
		else {
			start.SetBase16(left.c_str());
		}

		std::string right = s.substr(pos + 1);

		if (right[0] == '+') {
			Int t;
			t.SetBase16(right.substr(1).c_str());
			end.Set(&start);
			end.Add(&t);
		}
		else {
			end.SetBase16(right.c_str());
		}
	}

	return true;
}

#ifdef WIN64
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
        switch (fdwCtrlType) {
        case CTRL_C_EVENT:
                //printf("\n\nCtrl-C event\n\n");
                should_exit = true;
                externalStopRequested = true;
                return TRUE;

        default:
                return TRUE;
        }
}
#else
void CtrlHandler(int signum) {
        (void)signum;
        printf("\n\nStopping...\n");
        should_exit = true;
        externalStopRequested = 1;
}
#endif

int main(int argc, char** argv)
{
	// Global Init
	Timer::Init();
	rseed(Timer::getSeed32());

	int defaultCpuThreads = Timer::getCoreNumber();
#ifdef WITHGPU
	bool gpuEnable = true;
	int nbCPUThread = 0;
#else
	bool gpuEnable = false;
	int nbCPUThread = defaultCpuThreads;
#endif
        bool gpuAutoGrid = true;
        bool gpuUseOccupancyBlockSize = false;
	int compMode = SEARCH_COMPRESSED;
	vector<int> gpuId = { 0 };
	vector<int> gridSize;

	string outputFile = "Found.txt";

	string inputFile = "";	// for both multiple hash160s and x points
	string address = "";	// for single address mode
	string xpoint = "";		// for single x point mode

	std::vector<unsigned char> hashORxpoint;
	bool singleAddress = false;

	bool tSpecified = false;
	bool useSSE = true;
        uint32_t maxFound = 1024 * 64;

        uint64_t rKey = 0;

        uint32_t shardCount = 1;
        uint32_t shardIndex = 0;
        bool shardCountSpecified = false;
        bool shardIndexSpecified = false;
        std::string shardSummary;

        Int rangeStart;
        Int rangeEnd;
	rangeStart.SetInt32(0);
	rangeEnd.SetInt32(0);

	int searchMode = 0;
	int coinType = COIN_BTC;

	hashORxpoint.clear();

	// cmd args parsing
	CmdParse parser;
	parser.add("-h", "--help", false);
	parser.add("-c", "--check", false);
	parser.add("-l", "--list", false);
	parser.add("-u", "--uncomp", false);
	parser.add("-b", "--both", false);
        parser.add("-g", "--gpu", false);
        parser.add("", "--gpui", true);
        parser.add("", "--gpux", true);
#ifdef WITHGPU
        parser.add("", "--gpu-autoblock", false);
#endif
        parser.add("-t", "--thread", true);
	parser.add("-i", "--in", true);
	parser.add("-o", "--out", true);
        parser.add("-m", "--mode", true);
        parser.add("", "--coin", true);
        parser.add("", "--range", true);
        parser.add("", "--shard-total", true);
        parser.add("", "--shard-index", true);
        parser.add("-r", "--rkey", true);
        parser.add("-v", "--version", false);

	if (argc == 1) {
		usage();
		return 0;
	}
	try {
		parser.parse(argc, argv);
	}
	catch (std::string err) {
		printf("Error: %s\n", err.c_str());
		usage();
		exit(-1);
        }
        std::vector<OptArg> args = parser.getArgs();

        auto parseShardParameter = [](const std::string& text, const char* optionName) -> uint32_t {
                try {
                        size_t idx = 0;
                        unsigned long value = std::stoul(text, &idx, 10);
                        if (idx != text.size()) {
                                throw std::string(std::string(optionName) + " expects a non-negative integer value");
                        }
                        if (value > std::numeric_limits<uint32_t>::max()) {
                                throw std::string(std::string(optionName) + " value is too large");
                        }
                        return static_cast<uint32_t>(value);
                }
                catch (const std::exception&) {
                        throw std::string(std::string(optionName) + " expects a non-negative integer value");
                }
        };

        for (unsigned int i = 0; i < args.size(); i++) {
                OptArg optArg = args[i];
                std::string opt = args[i].option;

		try {
			if (optArg.equals("-h", "--help")) {
				usage();
				return 0;
			}
			else if (optArg.equals("-c", "--check")) {
				printf("KeyHunt-Cuda v" RELEASE "\n\n");
				printf("\nChecking... Secp256K1\n\n");
				Secp256K1* secp = new Secp256K1();
				secp->Init();
				secp->Check();
				printf("\n\nChecking... Int\n\n");
				Int* K = new Int();
				K->SetBase16("3EF7CEF65557B61DC4FF2313D0049C584017659A32B002C105D04A19DA52CB47");
				K->Check();
				delete secp;
				delete K;
				printf("\n\nChecked successfully\n\n");
				return 0;
			}
			else if (optArg.equals("-l", "--list")) {
#ifdef WIN64
				GPUEngine::PrintCudaInfo();
#else
				printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
				return 0;
			}
			else if (optArg.equals("-u", "--uncomp")) {
				compMode = SEARCH_UNCOMPRESSED;
			}
			else if (optArg.equals("-b", "--both")) {
				compMode = SEARCH_BOTH;
			}
			else if (optArg.equals("-g", "--gpu")) {
#ifdef WITHGPU
				gpuEnable = true;
				nbCPUThread = 0;
#else
				printf("Error: GPU support is not available, recompile with -DWITHGPU.\n");
				return -1;
#endif
			}
			else if (optArg.equals("", "--gpui")) {
				string ids = optArg.arg;
				getInts("--gpui", gpuId, ids, ',');
			}
                        else if (optArg.equals("", "--gpux")) {
                                string grids = optArg.arg;
                                getInts("--gpux", gridSize, grids, ',');
                                gpuAutoGrid = false;
                        }
#ifdef WITHGPU
                        else if (optArg.equals("", "--gpu-autoblock")) {
                                gpuUseOccupancyBlockSize = true;
                        }
#endif
                        else if (optArg.equals("-t", "--thread")) {
				nbCPUThread = std::stoi(optArg.arg);
				tSpecified = true;
				gpuEnable = false;
			}
			else if (optArg.equals("-i", "--in")) {
				inputFile = optArg.arg;
			}
			else if (optArg.equals("-o", "--out")) {
				outputFile = optArg.arg;
			}
			else if (optArg.equals("-m", "--mode")) {
				searchMode = parseSearchMode(optArg.arg);
			}
			else if (optArg.equals("", "--coin")) {
				coinType = parseCoinType(optArg.arg);
			}
                        else if (optArg.equals("", "--range")) {
                                std::string range = optArg.arg;
                                parseRange(range, rangeStart, rangeEnd);
                        }
                        else if (optArg.equals("", "--shard-total")) {
                                shardCount = parseShardParameter(optArg.arg, "--shard-total");
                                shardCountSpecified = true;
                        }
                        else if (optArg.equals("", "--shard-index")) {
                                shardIndex = parseShardParameter(optArg.arg, "--shard-index");
                                shardIndexSpecified = true;
                        }
                        else if (optArg.equals("-r", "--rkey")) {
                                rKey = std::stoull(optArg.arg);
                        }
			else if (optArg.equals("-v", "--version")) {
				printf("KeyHunt-Cuda v" RELEASE "\n");
				return 0;
			}
		}
		catch (std::string err) {
			printf("Error: %s\n", err.c_str());
			usage();
			return -1;
		}
	}

#ifdef WITHGPU
	if (!gpuEnable && !tSpecified) {
		nbCPUThread = defaultCpuThreads;
	}
#endif

	//
	if (coinType == COIN_ETH && (searchMode == SEARCH_MODE_SX || searchMode == SEARCH_MODE_MX/* || compMode == SEARCH_COMPRESSED*/)) {
		printf("Error: %s\n", "Wrong search or compress mode provided for ETH coin type");
		usage();
		return -1;
	}
	if (coinType == COIN_ETH) {
		compMode = SEARCH_UNCOMPRESSED;
		useSSE = false;
	}
	if (searchMode == (int)SEARCH_MODE_MX || searchMode == (int)SEARCH_MODE_SX)
		useSSE = false;


	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

	if (ops.size() == 0) {
		// read from file
		if (inputFile.size() == 0) {
			printf("Error: %s\n", "Missing arguments");
			usage();
			return -1;
		}
		if (searchMode != SEARCH_MODE_MA && searchMode != SEARCH_MODE_MX) {
			printf("Error: %s\n", "Wrong search mode provided for multiple addresses or xpoints");
			usage();
			return -1;
		}
	}
	else {
		// read from cmdline
		if (ops.size() != 1) {
			printf("Error: %s\n", "Wrong args or more than one address or xpoint are provided, use inputFile for multiple addresses or xpoints");
			usage();
			return -1;
		}
		if (searchMode != SEARCH_MODE_SA && searchMode != SEARCH_MODE_SX) {
			printf("Error: %s\n", "Wrong search mode provided for single address or xpoint");
			usage();
			return -1;
		}


		switch (searchMode) {
		case (int)SEARCH_MODE_SA:
		{
			address = ops[0];
			if (coinType == COIN_BTC) {
				if (address.length() < 30 || address[0] != '1') {
					printf("Error: %s\n", "Invalid address, must have Bitcoin P2PKH address or Ethereum address");
					usage();
					return -1;
				}
				else {
					if (DecodeBase58(address, hashORxpoint)) {
						hashORxpoint.erase(hashORxpoint.begin() + 0);
						hashORxpoint.erase(hashORxpoint.begin() + 20, hashORxpoint.begin() + 24);
						assert(hashORxpoint.size() == 20);
					}
				}
			}
			else {
				if (address.length() != 42 || address[0] != '0' || address[1] != 'x') {
					printf("Error: %s\n", "Invalid Ethereum address");
					usage();
					return -1;
				}
				address.erase(0, 2);
				for (int i = 0; i < 40; i += 2) {
					uint8_t c = 0;
					for (size_t j = 0; j < 2; j++) {
						uint32_t c0 = (uint32_t)address[i + j];
						uint8_t c2 = (uint8_t)strtol((char*)&c0, NULL, 16);
						if (j == 0)
							c2 = c2 << 4;
						c |= c2;
					}
					hashORxpoint.push_back(c);
				}
				assert(hashORxpoint.size() == 20);
			}
		}
		break;
		case (int)SEARCH_MODE_SX:
		{
			unsigned char xpbytes[32];
			xpoint = ops[0];
			Int* xp = new Int();
			xp->SetBase16(xpoint.c_str());
			xp->Get32Bytes(xpbytes);
			for (int i = 0; i < 32; i++)
				hashORxpoint.push_back(xpbytes[i]);
			delete xp;
			if (hashORxpoint.size() != 32) {
				printf("Error: %s\n", "Invalid xpoint");
				usage();
				return -1;
			}
		}
		break;
		default:
			printf("Error: %s\n", "Invalid search mode for single address or xpoint");
			usage();
			return -1;
			break;
		}
	}

#ifdef WITHGPU
        if (gpuUseOccupancyBlockSize) {
                if (!gpuEnable) {
                        printf("Error: --gpu-autoblock requires GPU acceleration to be enabled.\n");
                        usage();
                        return -1;
                }

                if (!gpuAutoGrid) {
                        printf("Warning: --gpu-autoblock overrides any --gpux values.\n");
                }

                gridSize.clear();
                for (size_t i = 0; i < gpuId.size(); i++) {
                        const int deviceId = gpuId.at(i);
                        int recommended = RecommendOccupancyBlockSizeForDevice(deviceId);
                        if (recommended <= 0) {
                                printf("Warning: Unable to determine occupancy-optimized block size for GPU %d, falling back to 128 threads per block.\n",
                                        deviceId);
                                recommended = 128;
                        }
                        else {
                                printf("Info: GPU %d occupancy-guided threads per block: %d\n", deviceId, recommended);
                        }
                        gridSize.push_back(-1);
                        gridSize.push_back(recommended);
                }
                gpuAutoGrid = true;
        }
#else
        if (gpuUseOccupancyBlockSize) {
                printf("Error: --gpu-autoblock requested but binary was built without GPU support. Recompile with -DWITHGPU.\n");
                usage();
                return -1;
        }
#endif

        if (gridSize.size() == 0) {
                for (int i = 0; i < gpuId.size(); i++) {
                        gridSize.push_back(-1);
                        gridSize.push_back(128);
                }
        }
        if (gridSize.size() != gpuId.size() * 2) {
                printf("Error: %s\n", "Invalid gridSize or gpuId argument, must have coherent size\n");
                usage();
                return -1;
        }

        if (shardIndexSpecified && !shardCountSpecified) {
                printf("Error: %s\n", "--shard-index requires --shard-total to be specified\n");
                usage();
                return -1;
        }
        if (shardCount == 0) {
                printf("Error: %s\n", "--shard-total must be at least 1\n");
                usage();
                return -1;
        }
        if (shardIndex >= shardCount) {
                printf("Error: %s\n", "--shard-index must be less than --shard-total\n");
                usage();
                return -1;
        }

        if (rangeStart.GetBitLength() <= 0) {
                printf("Error: %s\n", "Invalid start range, provide start range at least, end range would be: start range + 0xFFFFFFFFFFFFULL\n");
                usage();
                return -1;
        }

        if (shardCount > 1) {
                Int totalSpan(&rangeEnd);
                totalSpan.Sub(&rangeStart);

                Int shardCountInt;
                shardCountInt.SetInt32(shardCount);
                Int remainder;
                Int shardSize(&totalSpan);
                shardSize.Div(&shardCountInt, &remainder);

                const uint32_t remainderCount = remainder.GetInt32();
                const bool shardGetsExtra = (remainderCount > 0) && (shardIndex < remainderCount);
                if (shardSize.IsZero() && !shardGetsExtra) {
                        printf("Error: shard %u/%u does not cover any keys. Reduce the shard count or expand the range.\n",
                                shardIndex, shardCount);
                        return -1;
                }

                Int shardIndexInt;
                shardIndexInt.SetInt32(shardIndex);
                Int offset(&shardSize);
                offset.Mult(&shardIndexInt);

                if (remainderCount != 0) {
                        uint32_t extraForPrevious = std::min(shardIndex, remainderCount);
                        if (extraForPrevious != 0) {
                                Int extraInt;
                                extraInt.SetInt32(extraForPrevious);
                                offset.Add(&extraInt);
                        }
                }

                Int shardStart(&rangeStart);
                shardStart.Add(&offset);

                Int shardSpan(&shardSize);
                if (shardGetsExtra) {
                        Int one;
                        one.SetInt32(1);
                        shardSpan.Add(&one);
                }

                Int shardEnd(&shardStart);
                shardEnd.Add(&shardSpan);

                rangeStart.Set(&shardStart);
                rangeEnd.Set(&shardEnd);

                shardSummary = std::to_string(static_cast<uint64_t>(shardIndex) + 1u) + "/" +
                        std::to_string(static_cast<uint64_t>(shardCount));
        }
#ifdef WITHGPU
	if (gpuEnable && nbCPUThread > 0) {
		printf("Warning: CPU threads are disabled when GPU acceleration is active. Ignoring CPU thread setting.\n");
		nbCPUThread = 0;
	}
#endif

	// Let one CPU core free per gpu is gpu is enabled
	// It will avoid to hang the system
	if (!tSpecified && nbCPUThread > 1 && gpuEnable)
		nbCPUThread -= (int)gpuId.size();
	if (nbCPUThread < 0)
		nbCPUThread = 0;


	printf("\n");
	printf("KeyHunt-Cuda v" RELEASE "\n");
	printf("\n");
	if (coinType == COIN_BTC)
		printf("COMP MODE    : %s\n", compMode == SEARCH_COMPRESSED ? "COMPRESSED" : (compMode == SEARCH_UNCOMPRESSED ? "UNCOMPRESSED" : "COMPRESSED & UNCOMPRESSED"));
	printf("COIN TYPE    : %s\n", coinType == COIN_BTC ? "BITCOIN" : "ETHEREUM");
	printf("SEARCH MODE  : %s\n", searchMode == (int)SEARCH_MODE_MA ? "Multi Address" : (searchMode == (int)SEARCH_MODE_SA ? "Single Address" : (searchMode == (int)SEARCH_MODE_MX ? "Multi X Points" : "Single X Point")));
	bool useGpuDevice = gpuEnable;
	bool useCpuDevice = (nbCPUThread > 0);
	const char* deviceLabel;
	if (useGpuDevice && useCpuDevice) {
		deviceLabel = "CPU & GPU";
	}
	else if (useGpuDevice) {
		deviceLabel = "GPU";
	}
	else if (useCpuDevice) {
		deviceLabel = "CPU";
	}
	else {
		deviceLabel = "NONE";
	}
        printf("DEVICE       : %s\n", deviceLabel);
        if (!shardSummary.empty()) {
                printf("SHARD       : %s\n", shardSummary.c_str());
        }
        printf("CPU THREAD   : %d\n", nbCPUThread);
	if (gpuEnable) {
		printf("GPU IDS      : ");
		for (int i = 0; i < gpuId.size(); i++) {
			printf("%d", gpuId.at(i));
			if (i + 1 < gpuId.size())
				printf(", ");
		}
		printf("\n");
		printf("GPU GRIDSIZE : ");
		for (int i = 0; i < gridSize.size(); i++) {
			printf("%d", gridSize.at(i));
			if (i + 1 < gridSize.size()) {
				if ((i + 1) % 2 != 0) {
					printf("x");
				}
				else {
					printf(", ");
				}

			}
		}
		if (gpuAutoGrid)
			printf(" (Auto grid size)\n");
		else
			printf("\n");
	}
	printf("SSE          : %s\n", useSSE ? "YES" : "NO");
        printf("RKEY         : %" PRIu64 " Mkeys\n", rKey);
	printf("MAX FOUND    : %d\n", maxFound);
        if (coinType == COIN_BTC) {
                switch (searchMode) {
                case (int)SEARCH_MODE_MA:
                        printf("BTC HASH160s : %s\n", inputFile.c_str());
                        break;
		case (int)SEARCH_MODE_SA:
			printf("BTC ADDRESS  : %s\n", address.c_str());
			break;
		case (int)SEARCH_MODE_MX:
			printf("BTC XPOINTS  : %s\n", inputFile.c_str());
			break;
		case (int)SEARCH_MODE_SX:
			printf("BTC XPOINT   : %s\n", xpoint.c_str());
			break;
		default:
			break;
		}
	}
	else {
		switch (searchMode) {
		case (int)SEARCH_MODE_MA:
			printf("ETH ADDRESSES: %s\n", inputFile.c_str());
			break;
		case (int)SEARCH_MODE_SA:
			printf("ETH ADDRESS  : 0x%s\n", address.c_str());
			break;
		default:
			break;
		}
        }
        printf("OUTPUT FILE  : %s\n", outputFile.c_str());

        email::NotifyStartup(buildStartupSummary(RELEASE,
                coinType,
                compMode,
                searchMode,
                deviceLabel,
                useGpuDevice,
                useCpuDevice,
                nbCPUThread,
                gpuId,
                gridSize,
                gpuAutoGrid,
                gpuUseOccupancyBlockSize,
                useSSE,
                rKey,
                maxFound,
                inputFile,
                address,
                xpoint,
                outputFile,
                rangeStart.GetBase16(),
                rangeEnd.GetBase16(),
                shardSummary));


        auto computeGpuStepMultiplier = [](const std::vector<int>& gridSizeValues) {
                int multiplier = 1;
                for (size_t idx = 0; idx + 1 < gridSizeValues.size(); idx += 2) {
                        const int threadsPerGroup = gridSizeValues[idx + 1];
                        if (threadsPerGroup <= 0) {
                                continue;
                        }
                        int power = 1;
                        while (power < threadsPerGroup) {
                                const int next = power << 1;
                                if (next > threadsPerGroup) {
                                        break;
                                }
                                power = next;
                        }
                        if (power > multiplier) {
                                multiplier = power;
                        }
                }
                return multiplier;
        };

        const int gpuStepMultiplier = (gpuEnable && !gridSize.empty()) ? computeGpuStepMultiplier(gridSize) : 1;

#ifdef WIN64
        if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {
                KeyHunt* v;
                switch (searchMode) {
                case (int)SEARCH_MODE_MA:
                case (int)SEARCH_MODE_MX:
                        v = new KeyHunt(inputFile, compMode, searchMode, coinType, gpuEnable, outputFile, useSSE,
                                maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit, gpuStepMultiplier);
                        break;
                case (int)SEARCH_MODE_SA:
                case (int)SEARCH_MODE_SX:
                        v = new KeyHunt(hashORxpoint, compMode, searchMode, coinType, gpuEnable, outputFile, useSSE,
                                maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit, gpuStepMultiplier);
                        break;
                default:
                        printf("\n\nNothing to do, exiting\n");
                        return 0;
                        break;
                }
                v->Search(nbCPUThread, gpuId, gridSize, should_exit);
                delete v;
                email::NotifyShutdown(buildShutdownSummary());
                printf("\n\nBYE\n");
                return 0;
        }
        else {
                printf("Error: could not set control-c handler\n");
                return -1;
        }
#else
        signal(SIGINT, CtrlHandler);
        KeyHunt* v;
        switch (searchMode) {
        case (int)SEARCH_MODE_MA:
        case (int)SEARCH_MODE_MX:
                v = new KeyHunt(inputFile, compMode, searchMode, coinType, gpuEnable, outputFile, useSSE,
                        maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit, gpuStepMultiplier);
                break;
        case (int)SEARCH_MODE_SA:
        case (int)SEARCH_MODE_SX:
                v = new KeyHunt(hashORxpoint, compMode, searchMode, coinType, gpuEnable, outputFile, useSSE,
                        maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit, gpuStepMultiplier);
                break;
        default:
                printf("\n\nNothing to do, exiting\n");
                return 0;
                break;
        }
        v->Search(nbCPUThread, gpuId, gridSize, should_exit);
        delete v;
        email::NotifyShutdown(buildShutdownSummary());
        return 0;
#endif
}
