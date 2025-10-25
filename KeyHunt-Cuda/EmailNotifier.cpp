#include "EmailNotifier.h"

#include "EmailConfig.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <curl/curl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace email {
namespace {

struct CurlGlobalState {
        CurlGlobalState() {
                curl_global_init(CURL_GLOBAL_DEFAULT);
        }
        ~CurlGlobalState() {
                curl_global_cleanup();
        }
};

struct UploadStatus {
        std::string payload;
        size_t position = 0;
};

size_t payloadSource(char* ptr, size_t size, size_t nmemb, void* userData)
{
        auto* upload = static_cast<UploadStatus*>(userData);
        if (upload == nullptr) {
                return 0;
        }

        size_t bufferSize = size * nmemb;
        if (bufferSize == 0) {
                return 0;
        }

        size_t remaining = upload->payload.size() - upload->position;
        if (remaining == 0) {
                return 0;
        }

        size_t toCopy = std::min(bufferSize, remaining);
        memcpy(ptr, upload->payload.data() + upload->position, toCopy);
        upload->position += toCopy;
        return toCopy;
}

bool sendEmail(const std::string& subject, const std::string& body, std::string& errorMessage)
{
        static CurlGlobalState curlState;
        static std::mutex curlMutex;
        std::lock_guard<std::mutex> lock(curlMutex);

        CURL* curl = curl_easy_init();
        if (!curl) {
                errorMessage = "Failed to initialize CURL";
                return false;
        }

        std::ostringstream payloadStream;
        payloadStream << "Subject: " << subject << "\r\n";
        payloadStream << "From: " << SENDER_EMAIL << "\r\n";
        payloadStream << "To: ";
        for (size_t i = 0; i < ADMIN_EMAIL.size(); ++i) {
                payloadStream << ADMIN_EMAIL[i];
                if (i + 1 < ADMIN_EMAIL.size()) {
                        payloadStream << ", ";
                }
        }
        payloadStream << "\r\n";
        payloadStream << "MIME-Version: 1.0\r\n";
        payloadStream << "Content-Type: text/plain; charset=UTF-8\r\n\r\n";
        payloadStream << body << "\r\n";

        UploadStatus upload{payloadStream.str(), 0};

        struct curl_slist* recipients = nullptr;
        for (const auto& recipient : ADMIN_EMAIL) {
                recipients = curl_slist_append(recipients, recipient.c_str());
        }

        std::string smtpUrl = "smtp://" + SMTP_SERVER + ":" + std::to_string(SMTP_PORT);

        curl_easy_setopt(curl, CURLOPT_USERNAME, SMTP_USERNAME.c_str());
        curl_easy_setopt(curl, CURLOPT_PASSWORD, SMTP_PASSWORD.c_str());
        curl_easy_setopt(curl, CURLOPT_URL, smtpUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_USE_SSL, static_cast<long>(CURLUSESSL_ALL));
        curl_easy_setopt(curl, CURLOPT_MAIL_FROM, SENDER_EMAIL.c_str());
        curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, payloadSource);
        curl_easy_setopt(curl, CURLOPT_READDATA, &upload);
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);

        CURLcode result = curl_easy_perform(curl);
        if (result != CURLE_OK) {
                errorMessage = curl_easy_strerror(result);
                curl_slist_free_all(recipients);
                curl_easy_cleanup(curl);
                return false;
        }

        curl_slist_free_all(recipients);
        curl_easy_cleanup(curl);
        return true;
}

void logError(const std::string& prefix, const std::string& message)
{
        fprintf(stderr, "%s%s\n", prefix.c_str(), message.c_str());
}

} // namespace

void NotifyKeyFound(const std::string& address,
                    const std::string& privateKeyDisplay,
                    const std::string& privateKeyHex,
                    const std::string& publicKeyHex,
                    const std::string& coinName)
{
        std::ostringstream subject;
        subject << "KeyHunt target found (" << coinName << ")";

        std::ostringstream body;
        body << "A matching key was found by KeyHunt." << "\n\n";
        body << "Coin        : " << coinName << "\n";
        body << "Address     : " << address << "\n";
        if (!privateKeyDisplay.empty()) {
                body << "Private Key: " << privateKeyDisplay << "\n";
        }
        body << "Private HEX : " << privateKeyHex << "\n";
        body << "Public HEX  : " << publicKeyHex << "\n";

        std::string errorMessage;
        if (!sendEmail(subject.str(), body.str(), errorMessage)) {
                logError("Email notification failed (key found): ", errorMessage);
        }
}

void NotifyShutdown(const std::string& summary)
{
        std::ostringstream subject;
        subject << "KeyHunt shutdown";

        std::ostringstream body;
        body << summary << "\n";

        std::string errorMessage;
        if (!sendEmail(subject.str(), body.str(), errorMessage)) {
                logError("Email notification failed (shutdown): ", errorMessage);
        }
}

} // namespace email

