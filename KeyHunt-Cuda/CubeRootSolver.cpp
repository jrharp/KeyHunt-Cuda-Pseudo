#include "CubeRootSolver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

CubeRootSolver::CubeRootSolver(Secp256K1& secpRef, const Int& start, const Int& end)
        : secp(secpRef)
{
        rangeStart.Set((Int*)&start);
        rangeEnd.Set((Int*)&end);
        rangeSize.Set((Int*)&rangeEnd);
        rangeSize.Sub((Int*)&rangeStart);
        order.Set(&secp.order);

        // Parameter selection guided by cube-root heuristic.
        double approxRange = rangeSize.ToDouble();
        if (!std::isfinite(approxRange) || approxRange < 1.0) {
                approxRange = 1.0;
        }

        double desiredTable = std::cbrt(approxRange);
        if (!std::isfinite(desiredTable) || desiredTable < 64.0) {
                desiredTable = 64.0;
        }
        if (desiredTable > 262144.0) {
                desiredTable = 262144.0;
        }
        tableSize = static_cast<size_t>(desiredTable);
        if (tableSize < 64) {
                tableSize = 64;
        }
        size_t pow2 = 1;
        while (pow2 < tableSize) {
                pow2 <<= 1;
        }
        tableSize = pow2;

        double desiredWalk = std::sqrt(approxRange / static_cast<double>(tableSize));
        if (!std::isfinite(desiredWalk) || desiredWalk < 64.0) {
                desiredWalk = 64.0;
        }
        if (desiredWalk > 8.0e6) {
                desiredWalk = 8.0e6;
        }
        walkLength = static_cast<uint64_t>(desiredWalk);
        if (walkLength < 64) {
                walkLength = 64;
        }
        maxIterations = std::max<uint64_t>(walkLength * 20, tableSize * 4);

        distBits = static_cast<int>(std::round(std::log2(desiredWalk)));
        if (distBits < 10) {
                distBits = 10;
        }
        if (distBits > 26) {
                distBits = 26;
        }
        distMask = (distBits >= 63) ? std::numeric_limits<uint64_t>::max() : ((1ULL << distBits) - 1ULL);

        const size_t stepCount = 32;
        steps.resize(stepCount);
        stepMask = stepCount - 1;

        Int denom;
        denom.SetInt64(static_cast<uint64_t>(walkLength));
        if (denom.IsZero()) {
                denom.SetInt32(1);
        }
        Int averageStep(&rangeSize);
        averageStep.Div(&denom);
        if (averageStep.IsZero()) {
                averageStep.SetInt32(1);
        }

        for (size_t i = 0; i < stepCount; ++i) {
                Step& step = steps[i];
                step.distance.Set(&averageStep);
                step.distance.Add(static_cast<uint64_t>(i + 1));
                step.modDistance.Set(&step.distance);
                step.modDistance.Mod(&order);
                Int tmp(&step.modDistance);
                step.delta = secp.ComputePublicKey(&tmp);
        }
}

bool CubeRootSolver::SolveForXPoint(const Point& target, Int& outKey)
{
        ensureTable();
        return runWild(target, outKey);
}

bool CubeRootSolver::isDistinguished(const Point& p) const
{
        if (distMask == 0) {
                return true;
        }
        return (p.x.bits64[0] & distMask) == 0;
}

size_t CubeRootSolver::stepIndex(const Point& p) const
{
        return static_cast<size_t>(p.x.bits64[0] & stepMask);
}

std::string CubeRootSolver::pointKey(const Point& p) const
{
        std::array<unsigned char, 32> xBytes{};
        std::array<unsigned char, 32> yBytes{};
        p.x.Get32Bytes(xBytes.data());
        p.y.Get32Bytes(yBytes.data());
        std::string key;
        key.reserve(64);
        key.assign(reinterpret_cast<const char*>(xBytes.data()), xBytes.size());
        key.append(reinterpret_cast<const char*>(yBytes.data()), yBytes.size());
        return key;
}

void CubeRootSolver::ensureTable()
{
        if (!tableReady) {
                buildTable();
                tableReady = true;
        }
}

void CubeRootSolver::buildTable()
{
        table.clear();
        table.reserve(tableSize * 2);

        Int segment(&rangeSize);
        Int denom;
        denom.SetInt64(static_cast<uint64_t>(tableSize));
        if (denom.IsZero()) {
                denom.SetInt32(1);
        }
        segment.Div(&denom);
        if (segment.IsZero()) {
                segment.SetInt32(1);
        }

        Int seedKey(&rangeStart);
        Int seedMod(&seedKey);
        seedMod.Mod(&order);
        Point seedPoint = secp.ComputePublicKey(&seedMod);

        for (size_t i = 0; i < tableSize; ++i) {
                Point walkPoint(seedPoint);
                Int actualKey(&seedKey);
                Int modKey(&seedMod);
                walk(walkPoint, actualKey, modKey);
                table.emplace(pointKey(walkPoint), actualKey);

                seedKey.Add(&segment);
                seedMod.Set(&seedKey);
                seedMod.Mod(&order);
                seedPoint = secp.ComputePublicKey(&seedMod);
        }
}

void CubeRootSolver::walk(Point& point, Int& actual, Int& mod) const
{
        for (uint64_t iter = 0; iter < maxIterations; ++iter) {
                if (isDistinguished(point)) {
                        return;
                }
                const size_t idx = stepIndex(point);
                actual.Add(const_cast<Int*>(&steps[idx].distance));
                mod.ModAddK1order((Int*)&steps[idx].modDistance);
                Point delta(steps[idx].delta);
                point = secp.AddDirect(point, delta);
        }
}

bool CubeRootSolver::runWild(const Point& startPoint, Int& outKey) const
{
        Point point(startPoint);
        Int offset(static_cast<uint64_t>(0));

        for (uint64_t iter = 0; iter < maxIterations; ++iter) {
                if (isDistinguished(point)) {
                        auto it = table.find(pointKey(point));
                        if (it != table.end()) {
                                Int candidate(&(it->second));
                                candidate.Sub(&offset);
                                if (candidate.IsNegative()) {
                                        candidate.Add(const_cast<Int*>(&order));
                                }
                                if (!candidate.IsLower(const_cast<Int*>(&rangeStart)) &&
                                    candidate.IsLower(const_cast<Int*>(&rangeEnd))) {
                                        outKey.Set(&candidate);
                                        return true;
                                }
                        }
                }

                const size_t idx = stepIndex(point);
                offset.Add(const_cast<Int*>(&steps[idx].distance));
                Point delta(steps[idx].delta);
                point = secp.AddDirect(point, delta);
        }

        return false;
}

