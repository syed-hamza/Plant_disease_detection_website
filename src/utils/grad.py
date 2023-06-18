class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        for module in self.model.modules():
            if isinstance(module, self.target_layer):
                module.register_backward_hook(self.save_gradient)
                break
        return self.model(x)

    def backward_pass(self, scores):
        self.model.zero_grad()
        scores.backward(retain_graph=True)

    def generate(self, input_image, class_idx):
        input_image.unsqueeze_(0)
        input_image.requires_grad_()
        self.model.zero_grad()

        scores = self.forward_pass(input_image)
        target_score = scores[:, class_idx]
        target_score.backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.target_layer.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)

        return cam
