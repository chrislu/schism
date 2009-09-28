
#ifndef SCM_CORE_IO_FILE_CORE_H_INCLUDED
#define SCM_CORE_IO_FILE_CORE_H_INCLUDED

namespace scm {
namespace io {

class file_core
{
public:
    typedef char                char_type;
    typedef scm::int64          size_type;

public:
    file_core();
    file_core(const file_core& rhs);
    virtual ~file_core();

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

	size_type					set_end_of_file();

private:

}; // class file_core

} // namepspace io
} // namepspace scm

#endif // SCM_CORE_IO_FILE_CORE_H_INCLUDED
