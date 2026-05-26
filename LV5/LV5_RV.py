from openni import openni2

openni2.initialize("C:/Program Files/OpenNI2/Redist") 

# Open the device
dev = openni2.Device.open_any()

# Create a depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()

# Grab a frame to test
frame = depth_stream.read_frame()
frame_data = frame.get_buffer_as_uint16()
print(f"Captured a frame! Data length: {len(frame_data)}")

# Clean up
depth_stream.stop()
openni2.unload()

