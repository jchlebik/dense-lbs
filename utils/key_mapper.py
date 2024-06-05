
from dataset_generator.dataset_generator import MNISTHelmholtz as dts

class KeyMapper:
    """
    A class that maps keys between the dataset and the model.
    """

    dts_keys_to_model_keys_dict = {key : key + "_batch" for key in dts.get_collate_keys()}
    model_keys_to_dts_keys_dict = {value : key for key, value in dts_keys_to_model_keys_dict.items()}

    @classmethod
    def get_dataset_keys(cls):
        """
        Returns the keys of the dataset.
        """
        return cls.dts_keys_to_model_keys_dict.keys()
    
    @classmethod
    def get_batch_keys(cls):
        """
        Returns the keys of the batch.
        """
        return cls.dts_keys_to_model_keys_dict.values()
        
    @classmethod
    def get_dataset_to_model_mapping(cls):
        """
        Returns the mapping from dataset keys to model keys.
        """
        return cls.dts_keys_to_model_keys_dict
    
    @classmethod
    def get_model_to_dataset_mapping(cls):
        """
        Returns the mapping from model keys to dataset keys.
        """
        return cls.model_keys_to_dts_keys_dict  
    
    @classmethod
    def get_sound_speed_key(cls):
        """
        Returns the key for the sound speed.
        """
        return dts.get_sound_speed_key()
    
    @classmethod
    def get_density_key(cls):
        """
        Returns the key for the density.
        """
        return dts.get_density_key()
    
    @classmethod
    def get_pml_key(cls):
        """
        Returns the key for the PML.
        """
        return dts.get_pml_key()
    
    @classmethod
    def get_source_key(cls):
        """
        Returns the key for the source.
        """
        return dts.get_source_key()
    
    @classmethod
    def get_sos_field_key(cls):
        """
        Returns the key for the SOS field.
        """
        return dts.get_sos_field_key()
    
    @classmethod
    def get_full_field_key(cls):
        """
        Returns the key for the full field.
        """
        return dts.get_full_field_key()
    
    @classmethod
    def get_density_field_key(cls):
        """
        Returns the key for the density field.
        """
        return dts.get_density_field_key() 
