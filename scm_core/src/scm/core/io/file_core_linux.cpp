
#include "file_core_linux.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

file_core_linux::file_core_linux()
{
}

file_core_linux::~file_core_linux()
{
}

bool
file_core_linux::open(const std::string&       file_path,
                      std::ios_base::openmode  open_mode,
                      bool                     disable_system_cache,
                      scm::uint32              read_write_buffer_size,
                      scm::uint32              read_write_asynchronous_requests)
{

    return (true);
}

bool
file_core_linux::is_open() const
{
    return (false);
}

void
file_core_linux::close()
{
}

file_core_linux::size_type
file_core_linux::read(char_type*const output_buffer,
                      size_type       num_bytes_to_read)
{
    return (0);
}

file_core_linux::size_type
file_core_linux::write(const char_type*const input_buffer,
                       size_type             num_bytes_to_write)
{
    return (0);
}

file_core_linux::size_type
file_core_linux::set_end_of_file()
{
    return (0);
}

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
