#pragma once

#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace email {

inline std::string trim(const std::string& value)
{
        const auto start = value.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
                return {};
        }

        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(start, end - start + 1);
}

inline const std::map<std::string, std::string>& getConfigMap()
{
        static std::once_flag once;
        static std::map<std::string, std::string> values;
        static std::exception_ptr loadError;

        std::call_once(once, []() {
                const char* configPathEnv = std::getenv("KEYHUNT_SMTP_CONF");
                const bool pathExplicit = configPathEnv != nullptr && *configPathEnv != '\0';
                const std::string configPath = pathExplicit ? configPathEnv : "keyhunt_email.conf";

                std::ifstream file(configPath);
                if (!file.is_open()) {
                        if (pathExplicit) {
                                std::ostringstream oss;
                                oss << "Failed to open SMTP config file: " << configPath;
                                loadError = std::make_exception_ptr(std::runtime_error(oss.str()));
                        }
                        return;
                }

                std::string line;
                while (std::getline(file, line)) {
                        auto trimmedLine = trim(line);
                        if (trimmedLine.empty() || trimmedLine[0] == '#') {
                                continue;
                        }

                        auto delimiterPos = trimmedLine.find('=');
                        if (delimiterPos == std::string::npos) {
                                continue;
                        }

                        auto key = trim(trimmedLine.substr(0, delimiterPos));
                        auto value = trim(trimmedLine.substr(delimiterPos + 1));
                        if (!key.empty()) {
                                values[key] = value;
                        }
                }
        });

        if (loadError) {
                std::rethrow_exception(loadError);
        }

        return values;
}

inline std::string getConfigValue(const char* name)
{
        const char* envValue = std::getenv(name);
        if (envValue != nullptr && *envValue != '\0') {
                return envValue;
        }

        const auto& config = getConfigMap();
        auto it = config.find(name);
        if (it != config.end() && !it->second.empty()) {
                return it->second;
        }

        return {};
}

inline std::string getConfigOrThrow(const char* name)
{
        auto value = getConfigValue(name);
        if (value.empty()) {
                std::ostringstream oss;
                oss << "Configuration value '" << name << "' must be set";
                throw std::runtime_error(oss.str());
        }

        return value;
}

inline std::string GetSmtpServer()
{
        auto value = getConfigValue("KEYHUNT_SMTP_SERVER");
        if (!value.empty()) {
                return value;
        }

        return "smtp.gmail.com";
}

inline long GetSmtpPort()
{
        auto value = getConfigValue("KEYHUNT_SMTP_PORT");
        if (!value.empty()) {
                char* end = nullptr;
                long port = std::strtol(value.c_str(), &end, 10);
                if (end != value.c_str() && *end == '\0' && port > 0) {
                        return port;
                }
        }

        return 587;
}

inline std::string GetSmtpUsername()
{
        return getConfigOrThrow("KEYHUNT_SMTP_USERNAME");
}

inline std::string GetSmtpPassword()
{
        return getConfigOrThrow("KEYHUNT_SMTP_PASSWORD");
}

inline std::string GetSenderEmail()
{
        auto value = getConfigValue("KEYHUNT_SMTP_SENDER");
        if (!value.empty()) {
                return value;
        }

        return GetSmtpUsername();
}

inline std::vector<std::string> GetAdminRecipients()
{
        auto value = getConfigValue("KEYHUNT_SMTP_RECIPIENTS");
        if (value.empty()) {
                return {};
        }

        std::vector<std::string> recipients;
        std::stringstream ss(value);
        std::string item;
        while (std::getline(ss, item, ',')) {
                auto start = item.find_first_not_of(" \t\n\r");
                auto end = item.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                        recipients.emplace_back(item.substr(start, end - start + 1));
                }
        }

        return recipients;
}

} // namespace email

