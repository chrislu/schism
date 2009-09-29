
#ifndef SCM_CORE_IO_FILE_CORE_LINUX_H_INCLUDED
#define SCM_CORE_IO_FILE_CORE_LINUX_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <vector>

#include <scm/core/pointer_types.h>

#include <scm/core/io/file_core.h>

namespace scm {
namespace io {

class file_core_linux : public file_core
{
public:
    file_core_linux();
    virtual ~file_core_linux();

    // file_core interface
    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode,
                                     bool                     disable_system_cache,
                                     scm::uint32              read_write_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests);
    bool                        is_open() const;
    void                        close();

    size_type                   read(char_type*const output_buffer,
                                     size_type       num_bytes_to_read);
    size_type                   write(const char_type*const input_buffer,
                                      size_type             num_bytes_to_write);

	size_type			        set_end_of_file();
    // end file_core interface

}; // class file_core_linux

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX

#endif // SCM_CORE_IO_FILE_CORE_LINUX_H_INCLUDED
