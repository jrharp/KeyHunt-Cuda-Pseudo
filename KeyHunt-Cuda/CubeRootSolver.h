#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "SECP256k1.h"

class CubeRootSolver {
public:
        CubeRootSolver(Secp256K1& secp, const Int& rangeStart, const Int& rangeEnd);

        bool SolveForXPoint(const Point& target, Int& outKey);

private:
        struct Step {
                Int distance;
                Int modDistance;
                Point delta;
        };

        bool isDistinguished(const Point& p) const;
        size_t stepIndex(const Point& p) const;
        std::string pointKey(const Point& p) const;

        void ensureTable();
        void buildTable();
        void walk(Point& point, Int& actual, Int& mod) const;
        bool runWild(const Point& startPoint, Int& outKey) const;

        Secp256K1& secp;
        Int rangeStart;
        Int rangeEnd;
        Int rangeSize;
        Int order;

        std::vector<Step> steps;
        size_t stepMask = 0;
        int distBits = 0;
        uint64_t distMask = 0;
        size_t tableSize = 0;
        uint64_t walkLength = 0;
        uint64_t maxIterations = 0;

        mutable bool tableReady = false;
        mutable std::unordered_map<std::string, Int> table;
};

