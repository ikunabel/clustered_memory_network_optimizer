# clustered_memory_network

To run the network you can use the run function in the main file. 

Depending on the simulation that you want to run you can change the type of input that is connected to the network. It is possible to run the network with either constant background rate or constant background rate in combination with rate modulated input.

To control the properties that are recorded from the network, you can change the post_proc argument, if set to NETWORK_NMDA_STATS properties like the CV, CC, and relative NMDA uptime are computed (voltage traces will be recorded), if set to CAPACITY only the spike trains are recorded. 

The recorded data will be saved to the out_data/ directory (you maybe need to create it). To open the recorded files you can use:

```
with open("file_path", "rb") as f:
    # either:
    data = json.load(f)
    # or:
    raw = pickle.load(f)
```

