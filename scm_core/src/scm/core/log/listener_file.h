
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_LISTENER_FILE_H_INCLUDED
#define SCM_CORE_LOG_LISTENER_FILE_H_INCLUDED

#include <string>
#include <fstream>

#include <scm/core/log/listener.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace log {

class message;

class __scm_export(core) listener_file : public listener
{
public:
    listener_file(const std::string& file_name, bool append = false);
    virtual ~listener_file();

    void                notify(const message& msg);

private:
    std::string         _file_name;
    std::ofstream       _file_stream;

}; // class listener_file

} // namespace log
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_LOG_LISTENER_FILE_H_INCLUDED
