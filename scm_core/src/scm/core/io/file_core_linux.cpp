
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "file_core_linux.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cassert>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

#include <scm/log.h>
#include <scm/core/math/math.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>
#include <scm/core/utilities/foreach.h>

namespace scm {
namespace io {
namespace detail {

class fd_wrapper
{
public:
    explicit fd_wrapper(int fd, const std::string& fn) : _fd(fd), _fn(fn) {}
    ~fd_wrapper() {
        if (0 != ::close(_fd)) {
            std::string ret_error;
            switch (errno) {
                case EBADF: ret_error.assign("invalid file descriptor"); break;
                case EINTR: ret_error.assign("close interrupted by signal"); break;
                case EIO:   ret_error.assign("I/O error"); break;
                default:    ret_error.assign("unknown error"); break;
            }
            scm::err() << log::error
                       << "fd_wrapper::close(): "
                       << "error closing file "
                       << "(" << ret_error << ")"
                       << " '" << _fn << "'"
                       << log::end;
        }
    }

public:
    const int           _fd;
    const std::string&  _fn;

private:
    fd_wrapper(const fd_wrapper&);
    fd_wrapper& operator=(const fd_wrapper&);
}; // class fd_wrapper

file_core_linux::handle
file_adopter(int fd, const std::string& fn)
{
    shared_ptr<fd_wrapper> fdw(new fd_wrapper(fd, fn));
    return (file_core_linux::handle(fdw, &fdw->_fd));
}

file_core_linux::handle
file_open(const std::string& fn, int open_flags, mode_t create_mode) {

    int fd = ::open64(fn.c_str(), open_flags, create_mode);

    if (fd > -1) {
        return (file_adopter(fd, fn));
    }
    else {
        std::string ret_error;
        switch (errno) {
            case EACCES:    ret_error.assign("requested access to the file is not allowed (insufficient permissions?)"); break;
            case EEXIST:    ret_error.assign("pathname already exists and O_CREAT and O_EXCL were used"); break;
            case EFAULT:    ret_error.assign("pathname points outside your accessible address space"); break;
            case EISDIR:    ret_error.assign("pathname refers to a directory and the access requested involved writing"); break;
            default:        ret_error.assign("unknown error"); break;
        }
        scm::err() << log::error
                   << "file_open(): "
                   << "error opening file "
                   << "(" << ret_error << ")"
                   << " '" << fn << "'"
                   << log::end;

        return (file_core_linux::handle());
    }
}

} // namespace detail


file_core_linux::file_core_linux()
  : file_core()
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
    using namespace boost::filesystem;

    path            input_file_path(file_path);
    path            complete_input_file_path(system_complete(input_file_path));

    bool            input_file_exists = exists(complete_input_file_path);
    std::string     input_root_path = complete_input_file_path.root_name().string();


    int    open_flags  = O_LARGEFILE; // yes, we mainly go through this pain for large files
    mode_t create_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

    if (   (open_mode & std::ios_base::in)
        && (open_mode & std::ios_base::out)) {
        open_flags |= O_RDWR;
    }
    else if (open_mode & std::ios_base::in) {
        open_flags |= O_RDONLY;
    }
    else if (open_mode & std::ios_base::out) {
        open_flags |= O_WRONLY;
    }
    else {
        scm::err() << log::error
                   << "file_core_linux::open(): "
                   << "illegal open mode (missing 'in' or 'out' specification) "
                   << std::hex << open_mode
                   << " on file '" << file_path << "'" << log::end;
        return (false);
    }

    if (input_file_exists) {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            // everything ok
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            open_flags |= O_TRUNC;
        }
        else {
            scm::err() << log::error
                       << "file_core_linux::open(): "
                       << "illegal open mode "
                       << std::hex << open_mode
                       << " on file '" << file_path << "'" << log::end;
            return (false);
        }
    }
    else {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            open_flags |= O_CREAT;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            open_flags |= (O_CREAT | O_TRUNC);
        }
        else {
            scm::err() << log::error
                       << "file_core_linux::open(): "
                       << "illegal open mode "
                       << std::hex << open_mode
                       << " on file '" << complete_input_file_path.string() << "'" << log::end;
            return (false);
        }
    }

    if (disable_system_cache) {
        //flags_and_attributes |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
        // TODO implement async I/O
    }

    // do open
    _file_handle = detail::file_open(complete_input_file_path.string(), open_flags, create_mode);

    if (!_file_handle) {
        scm::err() << log::error
                   << "file_core_linux::open(): "
                   << "error creating/opening file:  "
                   << "'" << complete_input_file_path.string() << "'" << log::end;

        return (false);
    }

    if (   open_mode & std::ios_base::ate
        || open_mode & std::ios_base::app) {

        _position = actual_file_size();
    }

    _file_path  = complete_input_file_path.string();
    _open_mode  = open_mode;
    _file_size  = actual_file_size();

    return (true);
}

bool
file_core_linux::is_open() const
{
    if (_file_handle) {
        return (((*_file_handle) > -1) ? true : false);
    }
    else {
        return (false);
    }
}

void
file_core_linux::close()
{
    if (is_open()) {
        // in async write mode truncate the file
    }
    reset_values();
}

file_core_linux::size_type
file_core_linux::read(void*       output_buffer,
                      offset_type start_position,
                      size_type   num_bytes_to_read)
{
    assert(is_open());
    assert(_position >= 0);

    using namespace scm;

    uint8*      output_byte_buffer  = reinterpret_cast<uint8*>(output_buffer);
    offset_type bytes_read          = 0;

    _position = start_position;

    if (num_bytes_to_read <= 0) {
        return (0);
    }

//    // non system buffered read operation
//    if (async_io_mode()) {
//        bytes_read = read_async(output_buffer, num_bytes_to_read);
//    }
//    // normal system buffered operation
//    else
    {
        ssize_t file_bytes_read = 0;

        file_bytes_read = ::pread64(*_file_handle, output_byte_buffer, num_bytes_to_read, _position);

        if (file_bytes_read == -1) {
            scm::err() << log::error
                       << "file_core_linux::read(): "
                       << "error reading from file " << _file_path << log::end;
            return (0);
        }

        if (file_bytes_read == 0) {
            // eof
            return (-1);
        }
        if (file_bytes_read <= num_bytes_to_read) {
            _position           += file_bytes_read;
            bytes_read           = file_bytes_read;
        }
        else {
            scm::err() << log::error
                       << "file_core_linux::read(): "
                       << "unknown error reading from file " << _file_path << log::end;
            return (0);
        }
    }

    assert(bytes_read > 0);

    return (bytes_read);
}

file_core_linux::size_type
file_core_linux::write(const void* input_buffer,
                       offset_type start_position,
                       size_type   num_bytes_to_write)
{
    assert(is_open());

    using namespace scm;

    if (_open_mode & std::ios_base::app) {
        _position = _file_size;
    }

    const uint8*    input_byte_buffer   = reinterpret_cast<const uint8*>(input_buffer);
    offset_type     bytes_written       = 0;

    _position = start_position;
//    // non system buffered read operation
//    if (async_io_mode()) {
//        //scm::err() << log::error
//        //           << "file_win::write(): "
//        //           << "file was opened for async read operations (async write not supported)" << log::end;
//        //return (0);
//        bytes_written = write_async(input_buffer, num_bytes_to_write);
//    }
//    // normal system buffered operation
//    else
    {
        ssize_t file_bytes_written  = 0;

        file_bytes_written = ::pwrite64(*_file_handle, input_byte_buffer, num_bytes_to_write, _position);

        if (file_bytes_written == -1) {
            scm::err() << log::error
                       << "file_core_linux::write(): "
                       << "error writing to file " << _file_path << log::end;
            return (0);
        }

        if (file_bytes_written <= num_bytes_to_write) {
            _position           += file_bytes_written;
            bytes_written        = file_bytes_written;
        }
        else {
            scm::err() << log::error
                       << "file_core_linux::write(): "
                       << "unknown error writing to file " << _file_path << log::end;
        }
    }

    return (bytes_written);
}

bool
file_core_linux::flush_buffers() const
{
    assert(is_open());

    return (true);//FlushFileBuffers(_file_handle.get()) == TRUE ? true : false);
}

file_core_linux::offset_type
file_core_linux::set_end_of_file()
{
    if (is_open()) {
        if (0 != ftruncate64(*_file_handle, _position)) {
//            std::string ret_error;
//            switch (errno) {
//                case EBADF: ret_error.assign("invalid file descriptor"); break;
//                case EINTR: ret_error.assign("close interupted by signal"); break;
//                case EIO:   ret_error.assign("I/O error"); break;
//                default:    ret_error.assign("unknown error"); break;
//            }
            scm::err() << log::error
                       << "file_core_linux::set_end_of_file(): "
                       << "error truncating end of file: "
                       << "position " << std::hex << _position << " file "
                       << _file_path << log::end;

            close();
            return (-1);
        }

        return (_position);
    }

    return (1);
}

file_core_linux::size_type
file_core_linux::actual_file_size() const
{
    assert(is_open());

    off64_t new_pos = lseek64(*_file_handle, 0, SEEK_END);

    if (new_pos < 0) {
        std::string ret_error;
        switch (errno) {
            case EBADF:     ret_error.assign("invalid file descriptor"); break;
            case EINVAL:    ret_error.assign("invalid direction specified on lseek"); break;
            case EOVERFLOW: ret_error.assign("overflow on returned offset"); break;
            case ESPIPE:    ret_error.assign("file descripor is associated with a pipe, socket, or FIFO"); break;
            default:        ret_error.assign("unknown error"); break;
        }
        scm::err() << log::error
                   << "file_core_linux::actual_file_size(): "
                   << "error retrieving file size "
                   << "(" << ret_error << ")"
                   << " on file '" << _file_path << "'" << log::end;
    }

    return (new_pos);
}

bool
file_core_linux::set_file_pointer(offset_type new_pos)
{
    assert(is_open());

    off64_t ret_pos = lseek64(*_file_handle, new_pos, SEEK_SET);

    if (ret_pos < 0) {
        std::string ret_error;
        switch (errno) {
            case EBADF:     ret_error.assign("invalid file descriptor"); break;
            case EINVAL:    ret_error.assign("invalid direction specified on lseek"); break;
            case EOVERFLOW: ret_error.assign("overflow on returned offset"); break;
            case ESPIPE:    ret_error.assign("file descripor is associated with a pipe, socket, or FIFO"); break;
            default:        ret_error.assign("unknown error"); break;
        }
        scm::err() << log::error
                   << "file_core_linux::set_file_pointer(): "
                   << "error setting file pointer to position "
                   << std::hex << new_pos
                   << "(" << ret_error << ")"
                   << " on file '" << _file_path << "'" << log::end;
        return (false);
    }

    return (true);
}

void
file_core_linux::reset_values()
{
    file_core::reset_values();

    _file_handle.reset();
}

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX

