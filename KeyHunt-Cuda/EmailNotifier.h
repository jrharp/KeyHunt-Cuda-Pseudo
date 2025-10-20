#pragma once

#include <string>

namespace email {

// Sends a notification e-mail describing a found key.
void NotifyKeyFound(const std::string& address,
                    const std::string& privateKeyDisplay,
                    const std::string& privateKeyHex,
                    const std::string& publicKeyHex,
                    const std::string& coinName);

// Sends a notification indicating the search stopped.
void NotifyShutdown(const std::string& summary);

} // namespace email

