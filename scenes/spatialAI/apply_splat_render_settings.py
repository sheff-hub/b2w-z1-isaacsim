"""Isaac Sim Script Editor (Window > Script Editor > Run): run once after dragging a spatialAI_*.usdz into a stage.

File > Open on the .usdz needs nothing: the NuRec render settings are stored in its root layer and Isaac applies them.
Dragging it into another stage adds it as a reference, and Isaac only applies render settings from the stage's own root
layer. Without them (tone mapping that passes the splat colours through as captured) the splat looks washed out. This
copies the renderSettings of every layer the stage uses into the running renderer. It lasts for the session; run it
again after reopening the stage.
"""
import carb.settings
import omni.usd

settings = carb.settings.get_settings()
for layer in omni.usd.get_context().get_stage().GetUsedLayers():
    for key, value in layer.customLayerData.get("renderSettings", {}).items():
        settings.set("/" + key.replace(":", "/"), value)
        print(f"{layer.identifier.rsplit('/', 1)[-1]}: /{key.replace(':', '/')} = {value}")
