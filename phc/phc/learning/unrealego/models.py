
def create_model(opt):
    print(opt.model)

    if opt.model == 'egoglass':
        from .egoglass_model import EgoGlassModel
        model = EgoGlassModel()

    elif opt.model == "unrealego_heatmap_shared":
        from .unrealego_heatmap_shared_model import UnrealEgoHeatmapSharedModel
        model = UnrealEgoHeatmapSharedModel()

    elif opt.model == 'unrealego_autoencoder':
        from .unrealego_autoencoder_model import UnrealEgoAutoEncoderModel
        model = UnrealEgoAutoEncoderModel()

    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model