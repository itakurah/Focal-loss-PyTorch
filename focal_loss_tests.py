import unittest
import torch

from focal_loss import FocalLoss


class TestFocalLoss(unittest.TestCase):

    def test_binary_focal_loss(self):
        """ Test the FocalLoss with binary classification. """
        criterion = FocalLoss(gamma=2, alpha=0.25, task_type='binary')
        inputs = torch.randn(16)  # Logits from the model (batch_size=16)
        targets = torch.randint(0, 2, (16,)).float()  # Binary ground truth (0 or 1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss(self):
        """ Test the FocalLoss with multi-class classification. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=[0.25] * num_classes, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, num_classes, (16,))  # Ground truth with integer class labels (0 to num_classes-1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_label_focal_loss(self):
        """ Test the FocalLoss with multi-label classification. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-label')
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, 2, (16, num_classes)).float()  # Multi-label ground truth (0 or 1 for each class)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_binary_focal_loss_no_alpha(self):
        """ Test the FocalLoss with binary classification without alpha. """
        criterion = FocalLoss(gamma=2, task_type='binary')
        inputs = torch.randn(16)  # Logits from the model (batch_size=16)
        targets = torch.randint(0, 2, (16,)).float()  # Binary ground truth (0 or 1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss_no_alpha(self):
        """ Test the FocalLoss with multi-class classification without alpha. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, num_classes, (16,))  # Ground truth with integer class labels (0 to num_classes-1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_label_focal_loss_no_alpha(self):
        """ Test the FocalLoss with multi-label classification without alpha. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, task_type='multi-label')
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, 2, (16, num_classes)).float()  # Multi-label ground truth (0 or 1 for each class)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_invalid_task_type(self):
        """ Test FocalLoss with an invalid task type """
        with self.assertRaises(ValueError):
            criterion = FocalLoss(gamma=2, task_type='invalid-task')
            inputs = torch.randn(16, 5)
            targets = torch.randint(0, 5, (16,))
            criterion(inputs, targets)


if __name__ == '__main__':
    unittest.main()
