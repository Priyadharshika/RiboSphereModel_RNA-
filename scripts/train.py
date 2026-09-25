decoder = DiffusionTransformer(
    num_channels=512,
    input_channels=10 * 3,
    num_layers=8,
    num_heads=8,
    mlp_factor=4,
    normalize_queries_and_keys=False,
    conditioning_type="cat",
    share_adaln=False,
    attention_backend="sdpa",
)

flow_matcher = ConditionalFlowMatcher(
    sigma=0.0,
    time_sampling_mode="uniform"
)

optimizer = torch.optim.AdamW(
    list(encoder_model.parameters())
    + list(quantizer.parameters())
    + list(decoder.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints"

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)

num_epochs = 150

best_val_loss = float("inf")
patience = 8
patience_counter = 0
learning_rate=1e-4


for epoch in range(num_epochs):

    # =====================================================
    # TRAINING
    # =====================================================

    encoder_model.train()
    decoder.train()
    quantizer.train()

    total_train_loss = 0.0

    for i, sample in enumerate(train_data):

        # [L,10,3] → [1,L,10,3]
        coordinates = sample.unsqueeze(0).to(device)

        # Random rotation augmentation
        coordinates_rotated = random_rotation(coordinates)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        token_ids, loss = ribosphere_training_step(
            coordinates=coordinates_rotated,
            encoder_model=encoder_model,
            quantizer=quantizer,
            decoder=decoder,
            flow_matcher=flow_matcher,
            drop_cond_p=0.0
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_train_loss += loss.item()

    # Average training loss
    train_loss = total_train_loss / len(train_data)


    # =====================================================
    # VALIDATION
    # =====================================================

    encoder_model.eval()
    decoder.eval()
    quantizer.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for i, sample in enumerate(val_data):

            # [L,10,3] → [1,L,10,3]
            coordinates = sample.unsqueeze(0).to(device)

            token_ids, val_loss = ribosphere_training_step(
                coordinates=coordinates,
                encoder_model=encoder_model,
                quantizer=quantizer,
                decoder=decoder,
                flow_matcher=flow_matcher,
                drop_cond_p=0.0
            )

            total_val_loss += val_loss.item()

    # Average validation loss
    val_loss = total_val_loss / len(val_data)


    # =====================================================
    # CHECK FOR IMPROVEMENT
    # =====================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        filename = (
        f"RiboSphere_"
        f"epoch({epoch + 1})_"
        f"sample({num_train_samples})_"
        f"lr({learning_rate})_"
        f"tr({train_loss:.6f})_"
        f"vl({val_loss:.6f})_"
        f"seed({SEED}).pt")

        checkpoint_path = os.path.join(checkpoint_dir,filename)
        
        # Save best model
        torch.save({
            "epoch": epoch + 1,

            "encoder":
                encoder_model.state_dict(),

            "decoder":
                decoder.state_dict(),

            "quantizer":
                quantizer.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "train_loss": train_loss,
            "val_loss": val_loss,

        }, checkpoint_path)

        print(f"✓ Best model saved: {filename}")

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{patience})"
        )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if patience_counter >= patience:

        print("\nEarly stopping triggered!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best epoch: {epoch + 1 - patience_counter}")

        break