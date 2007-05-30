
#include "console_output.h"

#include <scm_core/core/ptr_types.h>

using namespace scm::con;

console_out_stream::console_out_stream()
{
}

console_out_stream::~console_out_stream()
{
}

console_out_stream::connection_type console_out_stream::connect_output_listener(console_out_stream::stream_updated_signal_type::slot_function_type listener)
{
    return (_output_stream_updated_signal.connect(listener));
}

bool console_out_stream::disconnect_output_listener(console_out_stream::connection_type& listener_connection)
{
    listener_connection.disconnect();

    if (listener_connection.connected()) {
        return (false);
    }
    else {
        return (true);
    }
}

void console_out_stream::emit_stream_updated_signal()
{
    std::streampos          beg = _stream.tellg();
    _stream.seekg(0, std::ios_base::end);
    std::streampos          end = _stream.tellg();
    _stream.seekg(beg);

    std::streamoff          len = end - beg;

    core::scoped_array<char>  buf(new char[len + 1]);

    _stream.read(buf.get(), len);
    buf[len] = '\0';

    std::string             line;
    
    line.assign(buf.get());

    _stream.clear();

    _output_stream_updated_signal(line, *this);
}
