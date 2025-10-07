#key components list
important_list = [
    "progress bar",
    "sign-graphic",
    "ui-graphic"
]

#avatar list
avatar_list = [
    "avatar",
    "avatar-nonhuman",
    "chat box",
    "chat bubble",
    "nametag"
]

#informational list
informational_list = [
    "progress bar",
    "sign-graphic",
    "ui-graphic",
    "HUD",
    "menu",
    "sign-text",
    "ui-text",
]

#key components list
important_list = [
    "progress bar",
    "sign-graphic",
    "ui-graphic"
]

#interactables list
interactable_list = [
    "button",
    "interactable",
    "portal",
    "spawner",
    "target",
    "watch",
    "writing surface",
    "writing utensil"
]

#seating areas list
seating_area_list = [
    "campfire",
    "seat-single",
    "seat-multiple",
    "table"
]

#user safety list
user_safety_list = [
    "guardian",
    "out of bounds"
]

#user HUD 
user_hud_list = [
    "controller",
    "hand",
    "dashboard",
    "HUD",
    "pointer-target"
]

other_list = [
    "indicator-mute",
    "indicator-talking",
    "locomotion-target"
]

special_labels = ['progress bar', 'guardian', 'dashboard', 'out of bounds']#, 'ui-text', 'ui-graphic']

action_list = avatar_list + informational_list + interactable_list + seating_area_list + user_safety_list + user_hud_list


#Truth label parameters for each label
#frame, audio_server, x, y, z
parameters_dictionary = {
    "nametag": {"frame":True, "audio_server":True,"x":True, "y":True, "z":True},
    "avatars": {"frame":True, "audio_server":True,"x":True, "y":True, "z":True},

    "spatial": {"frame":False, "audio_server":True,"x":True, "y":True, "z":True},
    "others": {"frame":False, "audio_server":False,"x":False, "y":False, "z":False},

    "ocr": {"frame":True, "audio_server":True,"x":True, "y":True, "z":True},
    "gpt": {"frame":True, "audio_server":True,"x":True, "y":True, "z":True},
}


#dictionary that returns the string action key assciocated with label
action_dictionary = {
    "avatars": ["avatar", "avatar-nonhuman"], 
    "spatial":["portal","campfire","seat-single","seat-multiple","table","dashboard","guardian","out of bounds"],
    
    "nothing":["controller","hand"],
    "other":["indicator-mute","indicator-talking","locomotion-target"],
    
    "ocr":["button","spawner","target","watch","writing surface","writing utensil","sign-text","ui-text","chat box", "chat bubble"],
    "gpt":["interactable","progress bar", "menu","sign-graphic","ui-graphic","HUD",],
}

def get_semantic_label(technical_label, context=None):
    """
    Convert technical labels to human-friendly descriptions.
    
    Args:
        technical_label: The raw label from the detection system
        context: Optional additional context for the mapping
        
    Returns:
        str: A human-friendly label for speech synthesis
    """
    # Create a mapping from technical labels to speech-friendly versions
    semantic_mapping = {
        # Basic UI elements
        "sign-text": "sign",
        "sign-graphic": "poster",
        "ui-text": "notification",
        "ui-graphic": "icon",
        "interactable": "interactive element",
        "HUD": "heads-up display",
        
        # Avatar types
        "avatar": "person",
        "avatar-nonhuman": "character",
        
        # Indicators and status elements
        "indicator-mute": "you are currently muted",
        "progress bar": "loading bar",
        "locomotion-target": "destination",
        
        # World elements
        "seat-single": "chair",
        "seat-multiple": "couch or bench",
        "writing utensil": "pen or marker",
        "guardian": "boundary",
        "out of bounds": "out of bounds area",
        
        # Default case - return the original if no mapping exists
        # This ensures we don't lose any labels
    }
    
    # Process context-specific overrides
    # if context == "muted" and technical_label == "indicator-mute":
    #     return "you are muted"
    # elif context == "unmuted" and technical_label == "indicator-mute":
    #     return "you are unmuted"
    
    # Return the semantic version or the original if no mapping exists
    return semantic_mapping.get(technical_label, technical_label)


#used to retrieve asset id for premade audio in playcanvas
asset_dictionary = {
    # Nametag
    "nametag_neutral": { 
        "key": 216587137,
        "in_view": 216587140  
    },
    "nametag_sad": {
        "key": 216587135,
        "in_view": 216587142
    },
    "nametag_cheerful": {
        "key": 216587133,
        "in_view": 216587134
    },
    "nametag_fearful": {
        "key": 216587136,
        "in_view": 216587141
    },
    "nametag_urgent": {
        "key": 216587138,
        "in_view": 216587139
    },
    
    # Avatar
    "avatar_neutral": {
        "key": 216587120,
        "in_view": 216587111
    },
    "avatar_sad": {
        "key": 216587112,
        "in_view": 216587114
    },
    "avatar_cheerful": {
        "key": 216587116,
        "in_view": 216587117
    },
    "avatar_fearful": {
        "key": 216587118,
        "in_view": 216587115
    },
    "avatar_urgent": {
        "key": 216587119,
        "in_view": 216587113
    },
    
    # Avatar-nonhuman
    "avatar_nonhuman_neutral": {
        "key": 216587125,
        "in_view": 216587126
    },
    "avatar_nonhuman_sad": {
        "key": 216587128,
        "in_view": 216587123
    },
    "avatar_nonhuman_cheerful": {
        "key": 216587122,
        "in_view": 216587129
    },
    "avatar_nonhuman_fearful": {
        "key": 216587130,
        "in_view": 216587131
    },
    "avatar_nonhuman_urgent": {
        "key": 216587124,
        "in_view": 216587127
    },
    
    # Portal
    "portal_neutral": {
        "key": 216590404,
        "in_view": 216590403
    },
    "portal_sad": {
        "key": 216590408,
        "in_view": 216590407
    },
    "portal_cheerful": {
        "key": 216590401,
        "in_view": 216590399
    },
    "portal_fearful": {
        "key": 216590405,
        "in_view": 216590406
    },
    "portal_urgent": {
        "key": 216590400,
        "in_view": 216590402
    },
    
    # Campfire
    "campfire_neutral": {
        "key": 216590355,
        "in_view": 216590353
    },
    "campfire_sad": {
        "key": 216590350,
        "in_view": 216590356
    },
    "campfire_cheerful": {
        "key": 216590358,
        "in_view": 216590354
    },
    "campfire_fearful": {
        "key": 216590357,
        "in_view": 216590351
    },
    "campfire_urgent": {
        "key": 216590352,
        "in_view": 216590349
    },
    
    # Seat-single
    "seat_single_neutral": {
        "key": 216590427,
        "in_view": 216590423
    },
    "seat_single_sad": {
        "key": 216590424,
        "in_view": 216590428
    },
    "seat_single_cheerful": {
        "key": 216590420,
        "in_view": 216590421
    },
    "seat_single_fearful": {
        "key": 216590426,
        "in_view": 216590419
    },
    "seat_single_urgent": {
        "key": 216590422,
        "in_view": 216590425
    },
    
    # Seat-multiple
    "seat_multiple_neutral": {
        "key": 216590415,
        "in_view": 216590417
    },
    "seat_multiple_sad": {
        "key": 216590414,
        "in_view": 216590410
    },
    "seat_multiple_cheerful": {
        "key": 216590416,
        "in_view": 216590418
    },
    "seat_multiple_fearful": {
        "key": 216590412,
        "in_view": 216590413
    },
    "seat_multiple_urgent": {
        "key": 216590411,
        "in_view": 216590409
    },
    
    # Table
    "table_neutral": {
        "key": 216590431,
        "in_view": 216590429
    },
    "table_sad": {
        "key": 216590438,
        "in_view": 216590432
    },
    "table_cheerful": {
        "key": 216590430,
        "in_view": 216590433
    },
    "table_fearful": {
        "key": 216590435,
        "in_view": 216590437
    },
    "table_urgent": {
        "key": 216590434,
        "in_view": 216590436
    },
    
    # Dashboard
    "dashboard_neutral": {
        "key": 216590365,
        "in_view": 216590368
    },
    "dashboard_sad": {
        "key": 216590360,
        "in_view": 216590363
    },
    "dashboard_cheerful": {
        "key": 216590361,
        "in_view": 216590362
    },
    "dashboard_fearful": {
        "key": 216590364,
        "in_view": 216590366
    },
    "dashboard_urgent": {
        "key": 216590367,
        "in_view": 216590359
    },
    
    # Pointer-target
    "pointer_target_neutral": {
        "key": 216590395,
        "in_view": 216590397
    },
    "pointer_target_sad": {
        "key": 216590392,
        "in_view": 216590396
    },
    "pointer_target_cheerful": {
        "key": 216590393,
        "in_view": 216590391
    },
    "pointer_target_fearful": {
        "key": 216590398,
        "in_view": 216590394
    },
    "pointer_target_urgent": {
        "key": 216590390,
        "in_view": 216590389
    },
    
    # Guardian
    "guardian_neutral": {
        "key": 216590372,
        "in_view": 216590375
    },
    "guardian_sad": {
        "key": 216590373,
        "in_view": 216590369
    },
    "guardian_cheerful": {
        "key": 216590371,
        "in_view": 216590377
    },
    "guardian_fearful": {
        "key": 216590378,
        "in_view": 216590370
    },
    "guardian_urgent": {
        "key": 216590374,
        "in_view": 216590376
    },
    
    # Out of bounds
    "out_of_bounds_neutral": {
        "key": 216590386,
        "in_view": 216590381

    },
    "out_of_bounds_sad": {
        "key": 216590388,
        "in_view": 216590383
    },
    "out_of_bounds_cheerful": {
        "key": 216590387,
        "in_view": 216590385
    },
    "out_of_bounds_fearful": {
        "key": 216590380,
        "in_view": 216590379
    },
    "out_of_bounds_urgent": {
        "key": 216590382,
        "in_view": 216590384
    },
    
    # Controller
    "controller_neutral": {
        "key": 216587369,
        "in_view": 216587364
    },
    "controller_sad": {
        "key": 216587362,
        "in_view": 216587365
    },
    "controller_cheerful": {
        "key": 216587367,
        "in_view": 216587363
    },
    "controller_fearful": {
        "key": 216587370,
        "in_view": 216587361
    },
    "controller_urgent": {
        "key": 216587366,
        "in_view": 216587368
    },
    
    # Hand
    "hand_neutral": {
        "key": 216587371,
        "in_view": 216587378
    },
    "hand_sad": {
        "key": 216587374,
        "in_view": 216587375
    },
    "hand_cheerful": {
        "key": 216587372,
        "in_view": 216587377
    },
    "hand_fearful": {
        "key": 216587380,
        "in_view": 216587379
    },
    "hand_urgent": {
        "key": 216587373,
        "in_view": 216587376
    },
    
    # Indicator-mute
    "indicator_mute_neutral": {
        "key": 216590316,
        "in_view": 216590317
    },
    "indicator_mute_sad": {
        "key": 216590310,
        "in_view": 216590308
    },
    "indicator_mute_cheerful": {
        "key": 216590313,
        "in_view": 216590311
    },
    "indicator_mute_fearful": {
        "key": 216590312,
        "in_view": 216590314
    },
    "indicator_mute_urgent": {
        "key": 216590309,
        "in_view": 216590315
    },
    
    # Indicator-talking
    "indicator_talking_neutral": {
        "key": 216590318,
        "in_view": 216590325
    },
    "indicator_talking_sad": {
        "key": 216590319,
        "in_view": 216590326
    },
    "indicator_talking_cheerful": {
        "key": 216590324,
        "in_view": 216590322
    },
    "indicator_talking_fearful": {
        "key": 216590327,
        "in_view": 216590320
    },
    "indicator_talking_urgent": {
        "key": 216590323,
        "in_view": 216590321
    },
    
    # Locomotion-target
    "locomotion_target_neutral": {
        "key": 216590328,
        "in_view": 216590330

    },
    "locomotion_target_sad": {
        "key": 216590336,
        "in_view": 216590332
    },
    "locomotion_target_cheerful": {
        "key": 216590329,
        "in_view": 216590335
    },
    "locomotion_target_fearful": {
        "key": 216590337,
        "in_view": 216590331
    },
    "locomotion_target_urgent": {
        "key": 216590333,
        "in_view": 216590334
    },
    
    # Button
    "button_neutral": {
        "key": 216590212,
        "in_view": 216590208
    },
    "button_sad": {
        "key": 216590211,
        "in_view": 216590204
    },
    "button_cheerful": {
        "key": 216590205,
        "in_view": 216590207
    },
    "button_fearful": {
        "key": 216590213,
        "in_view": 216590209
    },
    "button_urgent": {
        "key": 216590206,
        "in_view": 216590210
    },
    
    # Spawner
    "spawner_neutral": {
        "key": 216590249,
        "in_view": 216590244
    },
    "spawner_sad": {
        "key": 216590251,
        "in_view": 216590250
    },
    "spawner_cheerful": {
        "key": 216590246,
        "in_view": 216590247
    },
    "spawner_fearful": {
        "key": 216590252,
        "in_view": 216590248
    },
    "spawner_urgent": {
        "key": 216590245,
        "in_view": 216590253
    },
    
    # Target
    "target_neutral": {
        "key": 216590256,
        "in_view": 216590259
    },
    "target_sad": {
        "key": 216590263,
        "in_view": 216590255
    },
    "target_cheerful": {
        "key": 216590254,
        "in_view": 216590257
    },
    "target_fearful": {
        "key": 216590261,
        "in_view": 216590260
    },
    "target_urgent": {
        "key": 216590262,
        "in_view": 216590258
    },
    
    # Watch
    "watch_neutral": {
        "key": 216590281,
        "in_view": 216590278
    },
    "watch_sad": {
        "key": 216590282,
        "in_view": 216590276
    },
    "watch_cheerful": {
        "key": 216590279,
        "in_view": 216590283
    },
    "watch_fearful": {
        "key": 216590280,
        "in_view": 216590277
    },
    "watch_urgent": {
        "key": 216590275,
        "in_view": 216590274
    },
    
    # Writing surface
    "writing_surface_neutral": {
        "key": 216590286,
        "in_view": 216590291
    },
    "writing_surface_sad": {
        "key": 216590287,
        "in_view": 216590285
    },
    "writing_surface_cheerful": {
        "key": 216590290,
        "in_view": 216590293
    },
    "writing_surface_fearful": {
        "key": 216590292,
        "in_view": 216590284
    },
    "writing_surface_urgent": {
        "key": 216590288,
        "in_view": 216590289
    },
    
    # Writing utensil
    "writing_utensil_neutral": {
        "key": 216590302,
        "in_view": 216590299
    },
    "writing_utensil_sad": {
        "key": 216590296,
        "in_view": 216590297
    },
    "writing_utensil_cheerful": {
        "key": 216590298,
        "in_view": 216590301
    },
    "writing_utensil_fearful": {
        "key": 216590294,
        "in_view": 216590300
    },
    "writing_utensil_urgent": {
        "key": 216590295,
        "in_view": 216590303
    },
    
    # Sign-text
    "sign_text_neutral": {
        "key": 216590234,
        "in_view": 216590238
    },
    "sign_text_sad": {
        "key": 216590240,
        "in_view": 216590242
    },
    "sign_text_cheerful": {
        "key": 216590237,
        "in_view": 216590239
    },
    "sign_text_fearful": {
        "key": 216590236,
        "in_view": 216590241
    },
    "sign_text_urgent": {
        "key": 216590235,
        "in_view": 216590243
    },
    
    # UI-text
    "ui_text_neutral": {
        "key": 216590271,
        "in_view": 216590267
    },
    "ui_text_sad": {
        "key": 216590272,
        "in_view": 216590269
    },
    "ui_text_cheerful": {
        "key": 216590268,
        "in_view": 216590270
    },
    "ui_text_fearful": {
        "key": 216590273,
        "in_view": 216590266
    },
    "ui_text_urgent": {
        "key": 216590264,
        "in_view": 216590265
    },
    
    # Chat box
    "chat_box_neutral": {
        "key": 216590215,
        "in_view": 216590216
    },
    "chat_box_sad": {
        "key": 216590222,
        "in_view": 216590219
    },
    "chat_box_cheerful": {
        "key": 216590218,
        "in_view": 216590214
    },
    "chat_box_fearful": {
        "key": 216590220,
        "in_view": 216590221
    },
    "chat_box_urgent": {
        "key": 216590223,
        "in_view": 216590217
    },
    
    # Chat bubble
    "chat_bubble_neutral": {
        "key": 216590226,
        "in_view": 216590228
    },
    "chat_bubble_sad": {
        "key": 216590230,
        "in_view": 216590231
    },
    "chat_bubble_cheerful": {
        "key": 216590232,
        "in_view": 216590227
    },
    "chat_bubble_fearful": {
        "key": 216590233,
        "in_view": 216590229
    },
    "chat_bubble_urgent": {
        "key": 216590225,
        "in_view": 216590224
    },
    
    # Interactable
    "interactable_neutral": {
        "key": 216587310,
        "in_view": 216587311

    },
    "interactable_sad": {
        "key": 216587306,
        "in_view": 216587313
    },
    "interactable_cheerful": {
        "key": 216587315,
        "in_view": 216587307
    },
    "interactable_fearful": {
        "key": 216587309,
        "in_view": 216587312
    },
    "interactable_urgent": {
        "key": 216587314,
        "in_view": 216587308
    },
    
    # Progress bar
    "progress_bar_neutral": {
        "key": 216587331,
        "in_view": 216587329
    },
    "progress_bar_sad": {
        "key": 216587335,
        "in_view": 216587336
    },
    "progress_bar_cheerful": {
        "key": 216587327,
        "in_view": 216587333
    },
    "progress_bar_fearful": {
        "key": 216587332,
        "in_view": 216587334
    },
    "progress_bar_urgent": {
        "key": 216587330,
        "in_view": 216587328
    },
    
    # Menu
    "menu_neutral": {
        "key": 216587316,
        "in_view": 216587317
    },
    "menu_sad": {
        "key": 216587325,
        "in_view": 216587318
    },
    "menu_cheerful": {
        "key": 216587324,
        "in_view": 216587322
    },
    "menu_fearful": {
        "key": 216587319,
        "in_view": 216587321
    },
    "menu_urgent": {
        "key": 216587323,
        "in_view": 216587320
    },
    
    # Sign-graphic
    "sign_graphic_neutral": {
        "key": 216587344,
        "in_view": 216587343
    },
    "sign_graphic_sad": {
        "key": 216587345,
        "in_view": 216587339
    },
    "sign_graphic_cheerful": {
        "key": 216587338,
        "in_view": 216587342

    },
    "sign_graphic_fearful": {
        "key": 216587337,
        "in_view": 216587340
    },
    "sign_graphic_urgent": {
        "key": 216587341,
        "in_view": 216587346
    },
    
    # UI-graphic
    "ui_graphic_neutral": {
        "key": 216587356,
        "in_view": 216587348
    },
    "ui_graphic_sad": {
        "key": 216587350,
        "in_view": 216587354
    },
    "ui_graphic_cheerful": {
        "key": 216587355,
        "in_view": 216587352
    },
    "ui_graphic_fearful": {
        "key": 216587349,
        "in_view": 216587351
    },
    "ui_graphic_urgent": {
        "key": 216587353,
        "in_view": 216587347
    },
    
    # HUD
    "hud_neutral": {
        "key": 216587297,
        "in_view": 216587305
    },
    "hud_sad": {
        "key": 216587304,
        "in_view": 216587299
    },
    "hud_cheerful": {
        "key": 216587303,
        "in_view": 216587302
    },
    "hud_fearful": {
        "key": 216587298,
        "in_view": 216587301
    },
    "hud_urgent": {
        "key": 216587296,
        "in_view": 216587300
    }
}


