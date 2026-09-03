import bpy


def delete_all(context, types: list):
    for o in context.scene.objects:
        if o.type in types:
            o.select_set(True)
        else:
            o.select_set(False)
    bpy.ops.object.delete()


def find_all(context, type):
    rst = []
    for o in context.scene.objects:
        if o.type == type:
            rst.append(o)
    return rst
