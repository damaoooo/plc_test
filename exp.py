import angr

filename = "/home/damaoooo/Downloads/OpenPLC_v3/webserver/core/openplc"

entry_offset = 0x1288
proj = angr.Project(filename, load_options={'auto_load_libs':True})
print(proj.arch)
state = proj.factory.call_state(entry_offset, angr.PointerWrapper(0x1000))
print(state.regs.ip)
simgr = proj.factory.simulation_manager(state)
simgr.run()
print(simgr.deadended)