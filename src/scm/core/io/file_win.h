
#ifndef SCM_IO_FILE_WIN_H_INCLUDED
#define SCM_IO_FILE_WIN_H_INCLUDED

#ifdef _WIN32

#include <string>

#include <windows.h>

#include <scm/core/io/file_base.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

class __scm_export(core) file_win : public file_base
{
public:
    file_win();
    file_win(const file_win& rhs);
    virtual ~file_win();

    file_win&                   operator=(const file_win& rhs);
    void                        swap(file_win& rhs);

    using file_base::swap;

    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache = true,
                                     scm::uint32              read_write_buffer_size = detail::default_cache_buffer_size);
    bool                        is_open() const;
    void                        close();

    size_type                   read(char_type* output_buffer,
                                     size_type  num_bytes_to_read);
    size_type                   write(const char_type* input_buffer,
                                      size_type        num_bytes_to_write);
	size_type					set_end_of_file();

protected:
    size_type                   actual_file_size() const;
    bool                        set_file_pointer(size_type new_pos);

    size_type                   floor_vss(const size_type in_val) const;
    size_type                   ceil_vss(const size_type in_val) const;

    void                        reset_values();

protected:
    scm::shared_ptr<void>       _file_handle;

}; // class file

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // WIN32
#endif // SCM_IO_FILE_WIN_H_INCLUDED
