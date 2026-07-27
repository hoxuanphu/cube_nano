# F Prime v4.1.0 reference contract placeholder.
#
# The repository currently carries a generated dictionary, not the F Prime
# source checkout. Native build integration must map these names to the typed
# component ports without changing the Python wire/golden contract.

component CloudPayload {
  sync input port commandIn: Fw.Cmd
  output port telemetryOut: Fw.Tlm
  output port eventOut: Fw.Log
}
