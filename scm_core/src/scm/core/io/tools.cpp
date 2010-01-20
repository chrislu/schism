
#include "tools.h"

#include <cassert>
#include <iostream>

#include <scm/log.h>

#include <scm/core/pointer_types.h>
#include <scm/core/io/file.h>

namespace scm {
namespace io {

bool
read_text_file(const std::string& in_file_path, std::string& out_file_string)
{
    scoped_ptr<file>    f(new file());

    if (!f->open(in_file_path, std::ios_base::in, false)) {
        scm::out() << scm::log_level(scm::logging::ll_error)
                   << "read_text_file(): error opening file "
                   << in_file_path << std::endl;
        return (false);
    }

    // reserve at least file_size characters in string
    out_file_string.resize(f->size());

    assert(out_file_string.capacity() >= f->size());

    if (f->read(&out_file_string[0], f->size()) != f->size()) {
        scm::out() << scm::log_level(scm::logging::ll_error)
                   << "read_text_file(): error reading from file "
                   << in_file_path 
                   << " (number of bytes attempted to read: " << f->size() << ")" << std::endl;
        return (false);
    }

    f->close();

    return (true);
}

} // namespace io
} // namespace scm
