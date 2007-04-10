
#ifndef SYSTEM_EXCEPTION_H_INCLUDED
#define SYSTEM_EXCEPTION_H_INCLUDED

#include <exception>
#include <stdexcept>
#include <string>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm
{
    namespace core
    {
        class __scm_export system_exception : public std::runtime_error
        {
        public:
            system_exception(const std::string&);
        };

    } // namespace core
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // SYSTEM_EXCEPTION_H_INCLUDED
