#################################################
#################### IMPORTS ####################
#################################################

import json
import torch
import torch.nn as nn
import wandb

from utils import lr_scheduler_impl, initialize_wandb


###################################################################
#################### CUSTOM TRAINING FUNCTIONS ####################
###################################################################

def validate(model, device, test_loader):
	val_correct = 0
	epoch_val_loss = 0
	val_total = 0
	for i, (images, labels) in enumerate(test_loader):
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)
	
		_, predicted = torch.max(outputs.data, 1)
		val_correct += (predicted == labels).sum().item()
		epoch_val_loss += loss.item()
		val_total += labels.size(0)

	epoch_val_error = 1.0 - (val_correct / val_total)
	epoch_val_loss = epoch_val_loss / val_total
	return epoch_val_error, epoch_val_loss

# The training loop
def train(net, device, optimizer, scheduler, criterion, train_loader, test_loader, epochs, model_name):
	model = net.to(device)
	total_step = len(train_loader)
	overall_step = 0
	for epoch in range(epochs):
		correct = 0
		epoch_loss = 0
		total = 0
		model.train()
		for i, (images, labels) in enumerate(train_loader):
			if scheduler is None:
				# Update LR according to Cosine Annealing Warmup (discussed further down)
				next_lr = lr_scheduler_impl(total_step, epoch, i)
				for g in optimizer.param_groups:
					g['lr'] = next_lr
			else:
				next_lr = scheduler.get_last_lr()[0]
			# Move tensors to configured device
			images = images.to(device)
			labels = labels.to(device)
			# Forward Pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			# Compute loss and error
			_, predicted = torch.max(outputs.data, 1)
			iter_correct = (predicted == labels).sum().item()
			correct += iter_correct
			epoch_loss += loss.item()
			total += labels.size(0)
			# Log per weight Update
			wandb.log(
				{
					"Iteration Training Loss": loss.item(),
					"Iteration Training Error": 1.0 - (iter_correct / labels.size(0)),
					"Iteration LR": next_lr,
					"Iteration": epoch*total_step + i
				}
			)
			# Backpropogation and SGD
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) % config.batch_size == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
		# log training metrics
		epoch_error = 1.0 - (correct / total)
		epoch_loss = epoch_loss / total
		wandb.log(
			{
				"Epoch Training Loss": epoch_loss,
				"Epoch Training Error": epoch_error,
				"Epoch LR": next_lr,
				"Epoch": epoch,
				"Iteration": (epoch + 1)*total_step
			}
		)

		################ Validation ################
		model.eval()
		with torch.no_grad():
			epoch_val_error, epoch_val_loss = validate(model, test_loader)
		if scheduler is not None:
			scheduler.step()
		# log validation metrics
		wandb.log(
			{
				"Validation Loss": epoch_val_loss,
				"Validation Error": epoch_val_error,
				"Epoch": epoch,
				"Iteration": (epoch + 1)*total_step
			}
		)
		print('Error of the network on the test images: {} %'.format(100 * epoch_val_error))

		if (epoch+1) % 3 == 0 and epoch+1 != epochs:
			checkpoint = "epoch" + str(epoch+1).zfill(4) + ".h5"
			save_name = root_dir + "models/" + model_name + "/" + checkpoint
			torch.save(model.state_dict(), save_name)
	