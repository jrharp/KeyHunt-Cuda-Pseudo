#pragma once

#include <optional>
#include <string>

namespace email {

// Sends a notification e-mail describing a found key.
void NotifyKeyFound(const std::string& address,
                    const std::string& privateKeyDisplay,
                    const std::string& privateKeyHex,
                    const std::string& publicKeyHex,
                    const std::string& coinName);

// Sends a notification when the application starts running.
void NotifyStartup(const std::string& summary);

// Sends a notification indicating the search stopped.
void NotifyShutdown(const std::string& summary);

// Sends a periodic status update indicating the application is still running.
void NotifyHourlyUpdate(double elapsedSeconds, const std::optional<uint64_t>& lastCompletedBlock);

} // namespace email

