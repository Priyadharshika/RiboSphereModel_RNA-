def ribosphere_training_step(
    coordinates,
    encoder_model,
    quantizer,
    decoder,
    flow_matcher,
    drop_cond_p=0.0
):

    # -----------------------------------------
    # Input
    # [B,L,10,3]
    # -----------------------------------------

    batch_size, sequence_length, num_atoms, _ = coordinates.shape
    
    # -----------------------------------------
    # 2. Encoder
    # -----------------------------------------

    encoder_states = encoder_model(
        coordinates
    )

    # [B,L,256]

    # -----------------------------------------
    # 3. FSQ
    # -----------------------------------------

    conditioning_states, token_ids = quantizer(
        encoder_states
    )

    # conditioning_states: [B,L,512]
    # token_ids: [B,L]

    # -----------------------------------------
    # 4. Random source
    # -----------------------------------------

    source_coordinates = torch.randn_like(
        coordinates
    )

    source_coordinates = (
        source_coordinates
        - source_coordinates.mean(
            dim=(1, 2),
            keepdim=True
        )
    )

    # -----------------------------------------
    # 5. Flow matching
    # -----------------------------------------

    (
        times,
        intermediate_coordinates,
        target_vector_field
    ) = flow_matcher.sample_flow(
        source_coordinates,
        coordinates
    )

    # -----------------------------------------
    # 6. Conditioning dropout
    # -----------------------------------------

    condition_mask = (
        torch.rand(
            batch_size,
            device=coordinates.device
        )
        > drop_cond_p
    )[:, None, None]

    conditioning_states = (
        conditioning_states
        * condition_mask
    )

    # -----------------------------------------
    # 7. Flatten coordinates for decoder
    # -----------------------------------------

    flattened_intermediate_coordinates = (
        intermediate_coordinates.reshape(
            batch_size,
            sequence_length,
            30
        )
    )

    # -----------------------------------------
    # 8. Decoder
    # -----------------------------------------

    predicted_vector_field = decoder(
        flattened_intermediate_coordinates,
        times,
        conditioning_states=conditioning_states
    )

    # [B,L,30]

    # -----------------------------------------
    # 9. Restore [B,L,10,3]
    # -----------------------------------------

    predicted_vector_field = (
        predicted_vector_field.reshape(
            batch_size,
            sequence_length,
            10,
            3
        )
    )

    # -----------------------------------------
    # 10. Flow matching loss
    # -----------------------------------------

    flow_loss = (
        (
            target_vector_field
            - predicted_vector_field
        ) ** 2
    ).mean()

    return token_ids, flow_loss