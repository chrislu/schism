
#ifndef SCM_CORE_IO_FILE_CORE_H_INCLUDED
#define SCM_CORE_IO_FILE_CORE_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/io/file.h>

namespace scm {
namespace io {

class file_core : boost::noncopyable
{
public:
    typedef file::size_type     size_type;
    typedef file::char_type     char_type;

public:
    file_core();
    virtual ~file_core();

    virtual bool                open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode,
                                     bool                     disable_system_cache,
                                     scm::uint32              read_write_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests) = 0;
    virtual bool                is_open() const = 0;
    virtual void                close() = 0;

    virtual size_type           read(char_type*const output_buffer,
                                     size_type       num_bytes_to_read) = 0;
    virtual size_type           write(const char_type*const input_buffer,
                                      size_type             num_bytes_to_write) = 0;

	virtual size_type			set_end_of_file() = 0;

}; // class file_core

} // namepspace io
} // namepspace scm

#endif // SCM_CORE_IO_FILE_CORE_H_INCLUDED
