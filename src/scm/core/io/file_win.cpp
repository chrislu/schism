
#if WIN32

#include "file_win.h"

#include <algorithm>
#include <limits>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/math/math.h>

namespace scm {
namespace io {

file_win::file_win()
  : file_base()
{
}

file_win::file_win(const file_win& rhs)
  : file_base(rhs),
    _file_handle(rhs._file_handle)
{
}

file_win::~file_win()
{
}

file_win&
file_win::operator=(const file_win& rhs)
{
    file_win tmp(rhs);

    swap(tmp);

    return (*this);
}

void
file_win::swap(file_win& rhs)
{
    file_base::swap(rhs);

    std::swap(_file_handle,         rhs._file_handle);
}

file_win::size_type
file_win::actual_file_size() const
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);

    LARGE_INTEGER   cur_size_li;

    if (GetFileSizeEx(_file_handle.get(), &cur_size_li) == 0) {
        throw std::ios_base::failure("large_file_device_windows<char_type>::seek(): error retrieving current file size");
    }

    return (static_cast<size_type>(cur_size_li.QuadPart));
}

bool
file_win::set_file_pointer(size_type new_pos)
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);

    LARGE_INTEGER   position_li;

    position_li.QuadPart = new_pos;

    if (   SetFilePointer(_file_handle.get(), position_li.LowPart, &position_li.HighPart, FILE_BEGIN) == INVALID_SET_FILE_POINTER
        && GetLastError() != NO_ERROR) {
        return (false);
    }

    return (true);
}

file_win::size_type
file_win::floor_vss(const file_win::size_type in_val) const
{
    return((in_val / _volume_sector_size) * _volume_sector_size);
}

file_win::size_type
file_win::ceil_vss(const file_win::size_type in_val) const
{
    return ( ((in_val / _volume_sector_size)
            + (in_val % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size);
}

void
file_win::reset_values()
{
    file_base::reset_values();

    _file_handle.reset();
}

bool
file_win::open(const std::string&       file_path,
               std::ios_base::openmode  open_mode,
               bool                     disable_system_cache,
               scm::uint32              read_write_buffer_size)
{
    using namespace boost::filesystem;

    path            input_file_path(file_path, native);
    path            complete_input_file_path(system_complete(input_file_path));

    bool            input_file_exists = exists(complete_input_file_path);
    std::string     input_root_path = complete_input_file_path.root_name();

    // translate open mode to access mode
    DWORD desired_access = 0;
    DWORD read_write_buffer_access = 0;

    if (open_mode & std::ios_base::in) {
        desired_access              |= GENERIC_READ;
        read_write_buffer_access    =  PAGE_READWRITE;
    }
    if (open_mode & std::ios_base::out) {
        desired_access              |= GENERIC_WRITE;
        read_write_buffer_access    =  PAGE_READWRITE;
    }

    // share mode
    DWORD share_mode = FILE_SHARE_READ;

    // translate open mode to creation modes
    DWORD creation_disposition = 0;

    if (input_file_exists) {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = OPEN_ALWAYS;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = TRUNCATE_EXISTING;
        }
        else {
            //throw std::ios_base::failure("large_file_device_windows<char_type>::open(): illegal open mode");
            return (false);
        }
    }
    else {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = CREATE_NEW;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = CREATE_NEW;
        }
        else {
            //throw std::ios_base::failure("large_file_device_windows<char_type>::open(): illegal open mode");
            return (false);
        }
    }

    // file attributes
    DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;

    if (disable_system_cache) {
        flags_and_attributes |= FILE_FLAG_NO_BUFFERING;

        // retrieve the sector size information
        DWORD   sectors_per_cluster;
        DWORD   bytes_per_sector;
        DWORD   free_clusters;
        DWORD   total_clusters;

        if (GetDiskFreeSpace(input_root_path.c_str(),
                             &sectors_per_cluster,
                             &bytes_per_sector,
                             &free_clusters,
                             &total_clusters) == FALSE) {
            //throw std::ios_base::failure("large_file_device_windows<char_type>::open(): error retrieving sector size information");
            return (false);
        }

        _volume_sector_size = bytes_per_sector;

        assert(_volume_sector_size != 0);

        // calculate the correct read write buffer size (round up to full multiple of bytes per sector)
        _rw_buffer_size = static_cast<scm::int32>(ceil_vss(read_write_buffer_size));

        assert(_rw_buffer_size % _volume_sector_size == 0);
        assert(read_write_buffer_access != 0);

        // allocate read write buffer using VirtualAlloc
        // this aligns the memory region to page sizes
        // this memory has to be deallocated using VirtualFree, so the deallocator to the smart pointer
        _rw_buffer.reset(static_cast<char_type*>(VirtualAlloc(0, _rw_buffer_size, MEM_COMMIT | MEM_RESERVE, read_write_buffer_access)),
                                                 boost::bind<BOOL>(VirtualFree, _1, 0, MEM_RELEASE));

        if (!_rw_buffer) {
            //throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::open(): error allocating read write buffer for no system buffering operation: ")
            //                             + complete_input_file_path.string());
            return (false);
        }
    }

    // use shared pointer to manage the close in case we miss it somehow
    _file_handle.reset(CreateFile(complete_input_file_path.string().c_str(),
                                  desired_access,
                                  share_mode,
                                  0,
                                  creation_disposition,
                                  flags_and_attributes,
                                  0),
                       boost::bind<BOOL>(CloseHandle, _1));

    if (_file_handle.get() == INVALID_HANDLE_VALUE) {
        //throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::open(): error creating/opening file ")
        //                             + complete_input_file_path.string());
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
file_win::is_open() const
{
    if (_file_handle) {
        return ((_file_handle.get() == INVALID_HANDLE_VALUE) ? false : true);
    }
    else {
        return (false);
    }
}

void
file_win::close()
{
    if (is_open()) {
        // if we were non system buffered, so i may be possible to
        // be too large because of volume sector size restrictions
        if (   _rw_buffer_size
            && _open_mode & std::ios_base::out) {
            if (_file_size != actual_file_size()) {

                _file_handle.reset();
                _file_handle.reset(CreateFile(_file_path.c_str(),
                                              GENERIC_WRITE,
                                              PAGE_READONLY,
                                              0,
                                              OPEN_EXISTING,
                                              FILE_ATTRIBUTE_NORMAL,
                                              0),
                                   boost::bind<BOOL>(CloseHandle, _1));
            }
            if (_file_handle.get() == INVALID_HANDLE_VALUE) {
                throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::close(): error opening file for truncating end: ")
                                             + _file_path);
            }
            set_file_pointer(_file_size);
            if (SetEndOfFile(_file_handle.get()) == 0) {
                throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::close(): error truncating end of file: ")
                                             + _file_path);
            }
        }
    }

    // reset read write buffer invoking VirtualFree
    reset_values();
}

file_win::size_type
file_win::read(char_type* output_buffer,
               size_type  num_bytes_to_read)
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);
    assert(_position >= 0);

    using namespace scm;

    uint8*      output_byte_buffer  = reinterpret_cast<uint8*>(output_buffer);
    int64       bytes_read          = 0;

    // non system buffered read operation
    if (_rw_buffer_size > 0) {

        // set read pointer to beginning
        int64           current_position_vss    = 0;
        int64           file_size_vss           = 0;

        // align current position to volume sector size
        current_position_vss    = floor_vss(_position);
        file_size_vss           = ceil_vss(_file_size);

        // calculate the bytes to read in multiples of the volume sector size
        int64   bytes_to_read_vss   = 0;

        bytes_to_read_vss  = ceil_vss(math::min((_position % _volume_sector_size) + num_bytes_to_read,
                                                _file_size - current_position_vss));

        while (bytes_to_read_vss > 0) {
            int64   buffer_bytes_to_read        = 0;
            int64   end_position_vss            = 0;
            int64   rw_buffer_read_offset       = 0;
            DWORD   buffer_bytes_read           = 0;

            // determine the amount of bytes to read to the buffer
            // note if not a complete buffer is filled, the remainder is still
            // a interger multiple of the volume sector size
            buffer_bytes_to_read        = math::clamp<int64>(bytes_to_read_vss, 0, _rw_buffer_size);
            end_position_vss            = current_position_vss + buffer_bytes_to_read;

            assert(buffer_bytes_to_read > 0);
            assert(_rw_buffer);

            bool read_beg_in_rw_buf =    current_position_vss >= _rw_buffered_start
                                      && current_position_vss <  _rw_buffered_end;
            bool read_end_in_rw_buf =    end_position_vss >  _rw_buffered_start
                                      && end_position_vss <= _rw_buffered_end;

            // test if read begin or end is in buffered range
            if (read_beg_in_rw_buf && read_end_in_rw_buf) {
                // we can read this stuff completely from the rw buffer
                buffer_bytes_read       = static_cast<DWORD>(end_position_vss - current_position_vss);
                rw_buffer_read_offset   = current_position_vss - _rw_buffered_start;

                // the rw buffer is left untouched
            }
            else if (!read_beg_in_rw_buf && read_end_in_rw_buf) {
                // the end position is in the buffer
                int64   bytes_cached        = end_position_vss - _rw_buffered_start;
                int64   bytes_left_to_read  = math::max(_rw_buffered_start - current_position_vss,
                                                       math::min(_rw_buffered_start, _rw_buffer_size - bytes_cached));
                int64   new_read_pos_vss    = _rw_buffered_start - bytes_left_to_read;

                assert(bytes_cached <= (std::numeric_limits<size_t>::max)());

                // copy the buffered to the end
                memmove(_rw_buffer.get() + bytes_left_to_read,
                        _rw_buffer.get(),
                        static_cast<size_t>(bytes_cached));

                if (!set_file_pointer(new_read_pos_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
                }

                // read the necessary data from disc
                if (ReadFile(_file_handle.get(),
                             _rw_buffer.get(),
                             static_cast<DWORD>(bytes_left_to_read),
                             &buffer_bytes_read,
                             0) == 0)
                {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                if (buffer_bytes_read != bytes_left_to_read) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                // set values for the copy operation
                buffer_bytes_read       = static_cast<DWORD>(buffer_bytes_read + bytes_cached);
                rw_buffer_read_offset   = bytes_left_to_read - (_rw_buffered_start - current_position_vss);

                // set the rw buffer region
                _rw_buffered_start = new_read_pos_vss;
                _rw_buffered_end   = _rw_buffered_start + ceil_vss(buffer_bytes_read);
            }
            else if (read_beg_in_rw_buf && !read_end_in_rw_buf) {
                // beginning position in the buffer
                int64   bytes_cached        = _rw_buffered_end - current_position_vss;
                int64   bytes_left_to_read  = math::max(end_position_vss - _rw_buffered_end,
                                                        math::min(file_size_vss - _rw_buffered_end,
                                                                  _rw_buffer_size - bytes_cached));
                int64   new_read_pos_vss    = current_position_vss + bytes_cached;

                assert(bytes_cached <= (std::numeric_limits<size_t>::max)());

                // copy the buffered to the end
                memmove(_rw_buffer.get(),
                        _rw_buffer.get() + (current_position_vss - _rw_buffered_start),
                        static_cast<size_t>(bytes_cached));

                if (!set_file_pointer(new_read_pos_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
                }

                // read the necessary data from disc
                if (ReadFile(_file_handle.get(),
                             _rw_buffer.get() + bytes_cached,
                             static_cast<DWORD>(bytes_left_to_read),
                             &buffer_bytes_read,
                             0) == 0)
                {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                // set values for the copy operation
                buffer_bytes_read       = static_cast<DWORD>(buffer_bytes_read + bytes_cached);
                rw_buffer_read_offset   = 0;

                // set the rw buffer region
                _rw_buffered_start = new_read_pos_vss - bytes_cached;
                _rw_buffered_end   = _rw_buffered_start + ceil_vss(buffer_bytes_read);
            }
            else {
                // now we know we have to read everything from disk so fill up our internal rw buffer completely...
                if (!set_file_pointer(current_position_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
                }

                if (ReadFile(_file_handle.get(),
                             _rw_buffer.get(),
                             _rw_buffer_size,
                             &buffer_bytes_read,
                             0) == 0)
                {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                // set the rw buffer region
                _rw_buffered_start = current_position_vss;
                _rw_buffered_end   = _rw_buffered_start + ceil_vss(buffer_bytes_read);
            }

            // check if eof reached
            if (buffer_bytes_read <= 0) {
                return ((bytes_read == 0) ? -1 : bytes_read);
            }

            // copy data from rw buffer to output
            int64   buf_read_offset     = (_position % _volume_sector_size) + rw_buffer_read_offset;
            int64   buf_read_length     = math::min(math::max<int64>(0, static_cast<int64>(buffer_bytes_read) - (buf_read_offset - rw_buffer_read_offset)),
                                          math::min(num_bytes_to_read - bytes_read,
                                                    _rw_buffer_size - buf_read_offset));

            assert(buf_read_length >= 0);
            assert(buf_read_offset >= 0);
            assert(buf_read_offset + buf_read_length <= _rw_buffer_size);
            assert(buf_read_length <= num_bytes_to_read);
            
            assert(buf_read_length <= (std::numeric_limits<size_t>::max)());

            CopyMemory(output_byte_buffer,
                       _rw_buffer.get() + buf_read_offset,
                       static_cast<size_t>(buf_read_length));

            bytes_to_read_vss        = (buffer_bytes_read < buffer_bytes_to_read) ? 0 : bytes_to_read_vss - buffer_bytes_read;
            output_byte_buffer      += buf_read_length;
            _position               += buf_read_length;
            current_position_vss    += buffer_bytes_read;
            bytes_read              += buf_read_length;
        }
    }
    // normal system buffered operation
    else {
        if (!set_file_pointer(_position)) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
        }

        DWORD   file_bytes_read       = 0;

        if (ReadFile(_file_handle.get(), output_byte_buffer, static_cast<DWORD>(num_bytes_to_read), &file_bytes_read, 0) == 0) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
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
            throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unknown error reading from file");
        }
    }

    assert(bytes_read > 0);

    return (bytes_read);
}

file_win::size_type
file_win::write(const char_type* input_buffer,
                size_type        num_bytes_to_write)
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);
    assert(_position >= 0);

    using namespace scm;

    if (_open_mode & std::ios_base::app) {
        _position = _file_size;
    }

    const uint8*    input_byte_buffer   = reinterpret_cast<const uint8*>(input_buffer);
    int64           bytes_written       = 0;

    // non system buffered read operation
    if (_rw_buffer_size > 0) {

        // set read pointer to beginning
        int64           write_beg_file_offset_vss;
        int64           write_sector_prefetch_size;

        write_beg_file_offset_vss       = (_position / _volume_sector_size) * _volume_sector_size;  // floor
        write_sector_prefetch_size      = _position % _volume_sector_size;

        // set file pointer to beginning of volume sector
        if (!set_file_pointer(write_beg_file_offset_vss)) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set file pointer to current position");
        }

        if (write_sector_prefetch_size > 0) {
            if (_position <= _file_size) {
                int64 current_pos = _position;

                seek(write_beg_file_offset_vss, std::ios_base::beg);

                // ok we need some data from the beginning of the sector
                if (read(_rw_buffer.get(), write_sector_prefetch_size) < write_sector_prefetch_size) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable read data from beginning of sector");
                }

                seek(current_pos, std::ios_base::beg);

                // reset the file pointer to the beginning of the volume sector
                if (!set_file_pointer(write_beg_file_offset_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set file pointer to current position");
                }
            }
            else {
                if (SetEndOfFile(_file_handle.get()) == 0) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set new end of file");
                }
            }
        }

        // calculate the bytes to write in multiples of the volume sector size
        int64   bytes_to_write_vss  = 0;

        bytes_to_write_vss = write_sector_prefetch_size + num_bytes_to_write;
        bytes_to_write_vss =  ((bytes_to_write_vss / _volume_sector_size)
                             + (bytes_to_write_vss % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size; // ceil

        while (bytes_to_write_vss > 0) {
            int64   buffer_bytes_to_write   = 0;
            int64   buffer_fill_start_off   = 0;
            int64   buffer_fill_length      = 0;
            DWORD   buffer_bytes_written    = 0;

            // determine the amount of bytes to write to the buffer
            // note if not a complete buffer is filled, the remainder is still
            // a interger multiple of the volume sector size
            buffer_bytes_to_write = math::clamp<int64>(bytes_to_write_vss, 0, _rw_buffer_size);
            buffer_fill_start_off = _position % _volume_sector_size;
            buffer_fill_length    = math::min(buffer_bytes_to_write - buffer_fill_start_off,
                                              num_bytes_to_write - bytes_written);

            assert(buffer_bytes_to_write > 0);
            assert(_rw_buffer);

            assert(buffer_fill_length <= (std::numeric_limits<size_t>::max)());

            // fill in the data to be written
            CopyMemory(_rw_buffer.get() + buffer_fill_start_off,
                       input_byte_buffer,
                       static_cast<size_t>(buffer_fill_length));

            if (WriteFile(_file_handle.get(),
                          _rw_buffer.get(),
                          static_cast<DWORD>(buffer_bytes_to_write),
                          &buffer_bytes_written,
                          0) == 0) {
                throw std::ios_base::failure("large_file_device_windows<char_type>::write(): error writing to file");
            }

            bytes_to_write_vss  -= buffer_bytes_to_write;
            input_byte_buffer   += buffer_fill_length;
            _position           += buffer_fill_length;
            bytes_written       += buffer_fill_length;
        }

        // ok now we need to know the real file size, not the size rounded
        // to integer multiple of the volume sector size at the end
        if (_file_size < _position) {
            _file_size = _position;
        }
    }
    // normal system buffered operation
    else {
        if (!set_file_pointer(_position)) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set file pointer to current position");
        }

        DWORD   file_bytes_written  = 0;

        if (WriteFile(_file_handle.get(),
                      input_byte_buffer,
                      static_cast<DWORD>(num_bytes_to_write),
                      &file_bytes_written,
                      0) == 0) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): error writing from file");
        }

        if (file_bytes_written <= num_bytes_to_write) {
            _position           += file_bytes_written;
            bytes_written        = file_bytes_written;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unknown error reading from file");
        }
    }

    return (bytes_written);
}

} // namespace io
} // namespace scm

#endif
