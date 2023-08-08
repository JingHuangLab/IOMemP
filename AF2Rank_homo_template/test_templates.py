import sys
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import re
import subprocess
from collections import namedtuple
from absl import logging
from typing import Mapping, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name to save everything under")
parser.add_argument("--target_list", nargs='*', help="List of target names to run")
parser.add_argument("--targets_file", default="", help="File with list of target names to run")
parser.add_argument("--recycles", type=int, default=1, help="Number of recycles when predicting")
parser.add_argument("--model_num", type=int, default=1, help="Which AF2 model to use")
parser.add_argument("--seed", type=int, default=0, help="RNG Seed")
parser.add_argument("--verbose", action='store_true', help="print extra")
parser.add_argument("--deterministic", action='store_true', help="make all data processing deterministic (no masking, etc.)")
parser.add_argument("--use_native", action='store_true', help="add the native structure as a decoy, and compare outputs against it")
parser.add_argument("--mask_sidechains", action='store_true', help="mask out sidechain atoms except for C-Beta")
parser.add_argument("--mask_sidechains_add_cb", action='store_true', help="mask out sidechain atoms except for C-Beta, and add C-Beta to glycines")
parser.add_argument("--seq_replacement", default='', help="Amino acid residue to fill the decoy sequence with. Default keeps target sequence")
parser.add_argument("--af2_dir", help="AlphaFold code and weights directory")
parser.add_argument("--decoy_dir", default="/n/home01/jroney/template_injection/decoy_set/", help="Rosetta decoy directory")
parser.add_argument("--output_dir", default="/n/ovchinnikov_lab/Lab/af2rank/outputs/", help="Rosetta decoy directory")
parser.add_argument("--tm_exec", help="TMScore executable")
parser.add_argument("--multiple_decoy",action='store_true', help="Use multiple decoys for each prediction")
parser.add_argument("--no_template",action='store_true', help="Do not use any decoys")
parser.add_argument("--use_precomputed_msas",action='store_true', help="Use precomputed MSAs")
parser.add_argument("--msa_dir",type=str, help="directory containing precomputed MSAs")

args = parser.parse_args()

sys.path.append(args.af2_dir)

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import parsers
from alphafold.data import pipeline

from alphafold.common import protein
from alphafold.common import residue_constants

from alphafold.data.tools import kalign
from alphafold.data import mmcif_parsing
from alphafold.data import templates


# helper functions

"""
Read in a PDB file from a path
"""
def pdb_to_string(pdb_file):
  lines = []
  for line in open(pdb_file,"r"):
    if line[:6] == "HETATM" and line[17:20] == "MSE":
      line = "ATOM  "+line[6:17]+"MET"+line[20:]
    if line[:4] == "ATOM":
      lines.append(line)
  return "".join(lines)

"""
Compute aligned RMSD between two corresponding sets of poitns
true -- set of reference points. Numpy array of dimension N x 3
pred -- set of predicted points, Numpy array of dimension N x 3
"""
def jnp_rmsd(true, pred):
  def kabsch(P, Q):
    V, S, W = jnp.linalg.svd(P.T @ Q, full_matrices=False)
    flip = jax.nn.sigmoid(-10 * jnp.linalg.det(V) * jnp.linalg.det(W))
    S = flip * S.at[-1].set(-S[-1]) + (1-flip) * S
    V = flip * V.at[:,-1].set(-V[:,-1]) + (1-flip) * V
    return V@W
  p = true - true.mean(0,keepdims=True)
  q = pred - pred.mean(0,keepdims=True)
  p = p @ kabsch(p,q)
  loss = jnp.sqrt(jnp.square(p-q).sum(-1).mean() + 1e-8)
  return float(loss)

"""
Create an AlphaFold model runner
name -- The name of the model to get the parameters from. Options: model_[1-5]
"""
def make_model_runner(name, recycles):
  cfg = config.model_config(name)      

  cfg.data.common.num_recycle = recycles
  cfg.model.num_recycle = recycles
  cfg.data.eval.num_ensemble = 1
  if args.deterministic:
    cfg.data.eval.masked_msa_replace_fraction = 0.0
    cfg.model.global_config.deterministic = True
  params = data.get_model_haiku_params(name, args.af2_dir + 'data/')

  return model.RunModel(cfg, params)

"""
Make a set of empty features for no-template evalurations
"""
def empty_placeholder_template_features(num_templates, num_res):
  return {
      'template_aatype': np.zeros(
          (num_templates, num_res,
           len(residue_constants.restypes_with_x_and_gap)), dtype=np.float32),
      'template_all_atom_masks': np.zeros(
          (num_templates, num_res, residue_constants.atom_type_num),
          dtype=np.float32),
      'template_all_atom_positions': np.zeros(
          (num_templates, num_res, residue_constants.atom_type_num, 3),
          dtype=np.float32),
      'template_domain_names': np.zeros([num_templates], dtype=object),
      'template_sequence': np.zeros([num_templates], dtype=object),
      'template_sum_probs': np.zeros([num_templates], dtype=np.float32),
  }

"""
Read precomputed MSA from a file
"""
def read_precomputed_msa(msa_out_path, msa_format, max_sto_sequences=None):
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
    return result

"""
Create a feature dictionary for input to AlphaFold
runner - The model runner being invoked. Returned from `make_model_runner`
sequence - The target sequence being predicted
templates - The template features being added to the inputs
seed - The random seed being used for data processing
"""
def make_processed_feature_dict(runner, sequence, name="test", templates=None, seed=0, use_precomputed_msas=False, msa_output_dir=None):
  feature_dict = {}
  feature_dict.update(pipeline.make_sequence_features(sequence, name, len(sequence)))

  if use_precomputed_msas:
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_result = read_precomputed_msa(uniref90_out_path, 'sto', max_sto_sequences=10000) # the max number of sequences in the MSA following the AlphaFold paper2.2
    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])

    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    jackhmmer_mgnify_result = read_precomputed_msa(mgnify_out_path, 'sto', max_sto_sequences=501)
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
    
    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    hhblits_bfd_uniclust_result = read_precomputed_msa(bfd_out_path, 'a3m')
    bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])

    feature_dict.update(pipeline.make_msa_features((uniref90_msa, bfd_msa, mgnify_msa)))
  else:  
    msa = pipeline.parsers.parse_a3m(">1\n%s" % sequence)

    feature_dict.update(pipeline.make_msa_features([msa]))

  if templates is not None:
    feature_dict.update(templates)
  else:
    feature_dict.update(empty_placeholder_template_features(num_templates=0, num_res=len(sequence)))


  processed_feature_dict = runner.process_features(feature_dict, random_seed=seed)

  return processed_feature_dict

"""
Package AlphFold's output into an easy-to-use dictionary
prediction_result - output from running AlphaFold on an input dictionary
processed_feature_dict -- The dictionary passed to AlphaFold as input. Returned by `make_processed_feature_dict`.
"""
def parse_results(prediction_result, processed_feature_dict):
  b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
  dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
  dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
  contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

  out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
        "plddt": prediction_result['plddt'],
        "pLDDT": prediction_result['plddt'].mean(),
        "dists": dist_mtx,
        "adj": contact_mtx}

  out.update({"pae": prediction_result['predicted_aligned_error'],
              "pTMscore": prediction_result['ptm']})
  return out


'''
Function used to add C-Beta to glycine resides
input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
output: 4th coord
'''
def extend(a,b,c, L,A,D):
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  bc = N(b-c)
  n = N(np.cross(b-a, bc))
  m = [bc,np.cross(n,bc),n]
  d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
  return c + sum([m*d for m,d in zip(m,d)])



def align_pdb_template_to_query(
    query_sequence: str,
    cif_path, 
    kalign_binary_path: str) -> Tuple[str, Mapping[int, int]]:
  """Aligns template from the mmcif_object to the query.

  This method performs alignment and returns the new sequence and
  mapping.

  Args:
    query_sequence: The sequence of the query.
    template_chain_id: .
    mmcif_object: A mmcif_object which holds the actual template data.
    kalign_binary_path: The path to a kalign executable.

  Returns:
    A tuple template_sequence,query_to_template_mapping) where:
    *template_sequence is the actual template sequence that was found in
      the mmcif_object.
    *query_to_template_mapping is the mapping from the query to the
      actual template found in the mmcif_object.

  Raises:
    QueryToTemplateAlignError:
    * If there was an error thrown by the alignment tool.
  """
  aligner = kalign.Kalign(binary_path=kalign_binary_path)

  # mmcif object
  hit_pdb_code = cif_path[-21:-17]
  template_chain_id = cif_path[-17]
  print("hit_pdb_code:", hit_pdb_code)
  print("template_chain_id:", template_chain_id)
  # Fail if we can't find the mmCIF file.
  cif_string = templates._read_file(cif_path)
  parsing_result = mmcif_parsing.parse(
      file_id=hit_pdb_code, mmcif_string=cif_string, catch_all_errors=True)
  mmcif_object = parsing_result.mmcif_object
  print('mmcif parsing erros:',
              parsing_result.errors)
  template_sequence = mmcif_object.chain_to_seqres.get(
      template_chain_id, '')

  # Sometimes the template chain id is unknown. But if there is only a single
  # sequence within the mmcif_object, it is safe to assume it is that one.
  if not template_sequence:
    if len(mmcif_object.chain_to_seqres) == 1:
      logging.info('Could not find %s in %s, but there is only 1 sequence, so '
                   'using that one.',
                   template_chain_id,
                   mmcif_object.file_id)
      template_sequence = list(mmcif_object.chain_to_seqres.values())[0]
    else:
      raise templates.QueryToTemplateAlignError(
          f'Could not find chain {template_chain_id} in {mmcif_object.file_id}. '
          'If there are no mmCIF parsing errors, it is possible it was not a '
          'protein chain.')

  try:
    parsed_a3m = parsers.parse_a3m(
        aligner.align([query_sequence,template_sequence]))
    aligned_query,aligned_template = parsed_a3m.sequences
  except Exception as e:
    raise templates.QueryToTemplateAlignError(
        'Could not align query %s to template %s (%s_%s). Error: %s' %
        (query_sequence,template_sequence, mmcif_object.file_id,
         template_chain_id, str(e)))

  logging.info('Aligned query: %s\nAligned template: %s',
               aligned_query,aligned_template)
  print(f'Aligned query:\n{aligned_query}\nAligned template:\n{aligned_template}')
  
  mapping = {}
  query_index = -1
  template_index = -1
  for aa_q, aa_t in zip(aligned_query, aligned_template):
    if aa_q != '-':
      query_index += 1
    if aa_t != '-':
      template_index += 1
    if aa_q != '-' and aa_t != '-':
      mapping[query_index] = template_index

  template_sequence =template_sequence.replace('-', '')
  return template_sequence, mapping, mmcif_object, hit_pdb_code, template_chain_id

def score_homo_template(target_seq, model_runner, name, pdb_paths, seed=args.seed):
  """
  Ingest a homologous structure, pass it to AlphaFold as a template, and extract the parsed output
  target_seq -- the sequence to be predicted
  decoy_prot -- the decoy structure to be injected as a template
  model_runner -- the model runner to execute
  name -- the name associated with this prediction
  pdb_paths -- the paths of template file in shape (N)
  """
  templates_all_atom_positions_array = []
  templates_all_atom_masks_array = []
  templates_aatype_array = []
  template_domain_names_array = []

  print(pdb_paths)
  for pdb_path in pdb_paths:

    template_sequence, mapping, mmcif_object, pdb_id, chain_id = align_pdb_template_to_query(target_seq, 
            cif_path=pdb_path, kalign_binary_path=
            '/apps/anaconda/anaconda3/envs/alphafold/bin/kalign')
      
    try:
      # Essentially set to infinity - we don't want to reject templates unless
      # they're really really bad.
      all_atom_positions, all_atom_mask = templates._get_atom_positions(
          mmcif_object, chain_id, max_ca_ca_distance=150.0)
    except (templates.CaDistanceError, KeyError) as ex:
      raise templates.NoAtomDataInTemplateError(
          'Could not get atom data (%s_%s): %s' % (pdb_id, chain_id, str(ex))
          ) from ex

    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_masks = np.split(all_atom_mask, all_atom_mask.shape[0])

    output_templates_sequence = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in target_seq:
      # Residues in the query_sequence that are not in the template_sequence:
      templates_all_atom_positions.append(
          np.zeros((residue_constants.atom_type_num, 3)))
      templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
      output_templates_sequence.append('-')
    
    mapping_offset = 0

    if args.mask_sidechains_add_cb:
      # residues where we have all of the key backbone atoms (N CA C)
      for k, v in mapping.items():
        template_index = v + mapping_offset
        templates_all_atom_positions[k][:5] = all_atom_positions[template_index][0][:5]
        templates_all_atom_masks[k][:5] = all_atom_masks[template_index][0][:5]

        if template_sequence[template_index] == "G":
          cb = extend(all_atom_positions[template_index][0][2],
                      all_atom_positions[template_index][0][0],
                      all_atom_positions[template_index][0][1],
                      1.522, 1.927, -2.143)
          # print(f"The {template_index}th residue is added CB {cb}, and the coordinates of N atom {all_atom_positions[template_index][0][0]}.")
          templates_all_atom_positions[k][3] = cb
          templates_all_atom_masks[k][3] = 1
        
        if len(args.seq_replacement) != 1:
          output_templates_sequence[k] = template_sequence[v]

      # # Alanine (AA with the lowest number of atoms) has 5 atoms (C, CA, CB, N, O).
      # if np.sum(templates_all_atom_masks) < 5:
      #   raise templates.TemplateAtomMaskAllZerosError(
      #       'Template all atom mask was all zeros: %s_%s. Residue range: %d-%d' %
      #       (pdb_id, chain_id, min(mapping.values()) + mapping_offset,
      #       max(mapping.values()) + mapping_offset))

      output_templates_sequence = ''.join(output_templates_sequence)

      templates_aatype = residue_constants.sequence_to_onehot(
          output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)[None]
      
      templates_all_atom_positions_array.append(templates_all_atom_positions)
      templates_all_atom_masks_array.append(templates_all_atom_masks)
      templates_aatype_array.append(templates_aatype)
      template_domain_names_array.append('None')
      
    else: # Default AF settings
      for k, v in mapping.items():
        template_index = v + mapping_offset
        templates_all_atom_positions[k] = all_atom_positions[template_index][0]
        templates_all_atom_masks[k] = all_atom_masks[template_index][0]
        output_templates_sequence[k] = template_sequence[v]

      # Alanine (AA with the lowest number of atoms) has 5 atoms (C, CA, CB, N, O).
      if np.sum(templates_all_atom_masks) < 5:
        raise templates.TemplateAtomMaskAllZerosError(
            'Template all atom mask was all zeros: %s_%s. Residue range: %d-%d' %
            (pdb_id, chain_id, min(mapping.values()) + mapping_offset,
            max(mapping.values()) + mapping_offset))

      output_templates_sequence = ''.join(output_templates_sequence)

      templates_aatype = residue_constants.sequence_to_onehot(
          output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)
      
      templates_all_atom_positions_array.append(templates_all_atom_positions)
      templates_all_atom_masks_array.append(templates_all_atom_masks)
      templates_aatype_array.append(templates_aatype)
      template_domain_names_array.append(f'{pdb_id.lower()}_{chain_id}'.encode())
    
  template = {
          'template_all_atom_positions': np.array(templates_all_atom_positions_array),
          'template_all_atom_masks': np.array(templates_all_atom_masks_array),
          'template_aatype': np.array(templates_aatype_array),
          'template_domain_names': np.array(template_domain_names_array),
      }
  
  print(template['template_all_atom_positions'].shape)
  print(template['template_all_atom_masks'].shape)
  print(template['template_aatype'].shape)
  print(template['template_domain_names'].shape, flush=True)

  if args.use_precomputed_msas:
    features = make_processed_feature_dict(model_runner, target_seq, name=name, templates=template, seed=seed, use_precomputed_msas=True, msa_output_dir=args.msa_dir)
  else:
    features = make_processed_feature_dict(model_runner, target_seq, name=name, templates=template, seed=seed)
  
  prediction_result = model_runner.predict(features, random_seed=seed)
  result = parse_results(prediction_result, features)
  mismatch = False
  return result, mismatch

"""
Ingest a decoy protein, pass it to AlphaFold as a template, and extract the parsed output
target_seq -- the sequence to be predicted
decoy_prot -- the decoy structure to be injected as a template
model_runner -- the model runner to execute
name -- the name associated with this prediction
"""
def score_decoy(target_seq, decoy_prot, model_runner, name):
  decoy_seq_in = "".join([residue_constants.restypes[x] for x in decoy_prot.aatype]) # the sequence in the decoy PDB file

  mismatch = False
  if decoy_seq_in == target_seq:
    assert jnp.all(decoy_prot.residue_index - 1 == np.arange(len(target_seq)))
  else: # case when template is missing some residues
    if args.verbose:
      print("Sequece mismatch: {}".format(name))
    mismatch=True

    assert "".join(target_seq[i-1] for i in decoy_prot.residue_index) == decoy_seq_in 
  
  # use this to index into the template features
  template_idxs = decoy_prot.residue_index-1
  template_idx_set = set(template_idxs)

  # The sequence associated with the decoy. Always has same length as target sequence.
  decoy_seq = args.seq_replacement*len(target_seq) if len(args.seq_replacement) == 1 else target_seq

  # create empty template features
  pos = np.zeros([1,len(decoy_seq), 37, 3])
  atom_mask = np.zeros([1, len(decoy_seq), 37])

  if args.mask_sidechains_add_cb:
    pos[0, template_idxs, :5] = decoy_prot.atom_positions[:,:5]

    # residues where we have all of the key backbone atoms (N CA C)
    backbone_modelled = jnp.all(decoy_prot.atom_mask[:,[0,1,2]] == 1, axis=1)
    backbone_idx_set = set(decoy_prot.residue_index[backbone_modelled] - 1)

    projected_cb = [i-1 for i,b,m in zip(decoy_prot.residue_index, backbone_modelled, decoy_prot.atom_mask) if m[3] == 0 and b]
    projected_cb_set = set(projected_cb)
    gly_idx = [i for i,a in enumerate(target_seq) if a == "G"]
    assert all([k in projected_cb_set for k in gly_idx if k in template_idx_set and k in backbone_idx_set]) # make sure we are adding CBs to all of the glycines

    cbs = np.array([extend(c,n,ca, 1.522, 1.927, -2.143) for c, n ,ca in zip(pos[0,:,2], pos[0,:,0], pos[0,:,1])])

    pos[0, projected_cb, 3] = cbs[projected_cb]
    atom_mask[0, template_idxs, :5] = decoy_prot.atom_mask[:, :5]
    atom_mask[0, projected_cb, 3] = 1

    template = {"template_aatype":residue_constants.sequence_to_onehot(decoy_seq, residue_constants.HHBLITS_AA_TO_ID)[None],
                "template_all_atom_masks": atom_mask,
                "template_all_atom_positions":pos,
                "template_domain_names":np.asarray(["None"])}
  elif args.mask_sidechains:
    pos[0, template_idxs, :5] = decoy_prot.atom_positions[:,:5]
    atom_mask[0, template_idxs, :5] = decoy_prot.atom_mask[:,:5]

    template = {"template_aatype":residue_constants.sequence_to_onehot(decoy_seq, residue_constants.HHBLITS_AA_TO_ID)[None],
                "template_all_atom_masks": atom_mask,
                "template_all_atom_positions":pos,
                "template_domain_names":np.asarray(["None"])}
  else:
    pos[0, template_idxs] = decoy_prot.atom_positions
    atom_mask[0, template_idxs] = decoy_prot.atom_mask

    template = {"template_aatype":residue_constants.sequence_to_onehot(decoy_seq, residue_constants.HHBLITS_AA_TO_ID)[None],
                "template_all_atom_masks":decoy_prot.atom_mask[None],
                "template_all_atom_positions":decoy_prot.atom_positions[None],
                "template_domain_names":np.asarray(["None"])}

  print(template['template_aatype'].shape)
  print(template['template_all_atom_masks'].shape)
  print(template['template_all_atom_positions'].shape)
  print(template['template_domain_names'].shape)

  features = make_processed_feature_dict(model_runner, target_seq, name=name, templates=template, seed=args.seed)
  result = parse_results(model_runner.predict(features, random_seed=args.seed), features)
  return result, mismatch


tm_re = re.compile(r'TM-score[\s]*=[\s]*(\d.\d+)')
ref_len_re = re.compile(r'Length=[\s]*(\d+)[\s]*\(by which all scores are normalized\)')
common_re = re.compile(r'Number of residues in common=[\s]*(\d+)')
super_re = re.compile(r'\(":" denotes the residue pairs of distance < 5\.0 Angstrom\)\\n([A-Z\-]+)\\n[" ", :]+\\n([A-Z\-]+)\\n')

"""
Compute TM Scores between two PDBs and parse outputs
pdb_pred -- The path to the predicted PDB
pdb_native -- The path to the native PDB
test_len -- run asserts that the input and output should have the same length
"""
def compute_tmscore(pdb_pred, pdb_native, test_len=True):
  cmd = ([args.tm_exec, pdb_pred, pdb_native])
  tmscore_output = str(subprocess.check_output(cmd))
  try:
    tm_out = float(tm_re.search(tmscore_output).group(1))
    # reflen = int(ref_len_re.search(tmscore_output).group(1))
    # common = int(common_re.search(tmscore_output).group(1))
    
    # seq1 = super_re.search(tmscore_output).group(1)
    # seq2 = super_re.search(tmscore_output).group(1)
  except Exception as e:
    print("Failed on: " + " ".join(cmd))
    raise e

  # if test_len:
  #   assert reflen == common, cmd
  #   assert seq1 == seq2, cmd
  #   assert len(seq1) == reflen, cmd

  return tm_out

# Simple wrapper for keeping track of the information associated with each decoy. 
decoy_fields_list = ['target', 'decoy_id', 'decoy_path', 'rmsd', 'rosettascore', 'gdt_ts', 'tmscore', 'danscore']
Decoy = namedtuple("Decoy", decoy_fields_list)


# headers for csv outputs
csv_headers = decoy_fields_list + ['output_path', 'rmsd_out', 'tm_diff', 'tm_out', 'plddt', 'ptm']

def write_results(decoy, af_result, prot_native=None, pdb_native=None, mismatch=False):
  plddt = float(af_result['pLDDT'])
  ptm = float(af_result["pTMscore"])
  if prot_native is None:
    rms_out = -1
  else:
    rms_out = jnp_rmsd(prot_native.atom_positions[:,1,:], af_result['unrelaxed_protein'].atom_positions[:,1,:])

  pdb_lines = protein.to_pdb(af_result["unrelaxed_protein"])
  pdb_out_path = args.output_dir + args.name + "/pdbs/" + decoy.target + "_" + decoy.decoy_id[:-4]+'.pdb'
  with open(pdb_out_path, 'w') as f:
    f.write(pdb_lines)

  if decoy.decoy_id != "none.pdb":
    tm_diff = compute_tmscore(decoy.decoy_path[:-4]+'.pdb', pdb_out_path, test_len = False)
  else:
    tm_diff = -1

  if pdb_native is None:
    tm_out = -1
  else:
    tm_out = compute_tmscore(pdb_out_path, pdb_native)

  if not os.path.exists(args.output_dir + args.name + "/results/results_{}.csv".format(decoy.target)):
    with open(args.output_dir + args.name + "/results/results_{}.csv".format(decoy.target), "w") as f:
      f.write(",".join(csv_headers) + "\n")


  with open(args.output_dir + args.name + "/results/results_{}.csv".format(decoy.target), "a") as f:
    result_fields = [str(x) for x in list(decoy) + [pdb_out_path, rms_out, tm_diff, tm_out, plddt, ptm]]
    f.write(",".join(result_fields) + "\n")

    if args.verbose:
      print(",".join([x + "=" + y for x,y in zip(csv_headers, result_fields)]))


# create all of the output directoryes
os.makedirs(args.output_dir + args.name, exist_ok=True)
os.makedirs(args.output_dir + args.name + "/pdbs", exist_ok=True)
os.makedirs(args.output_dir + args.name + "/results", exist_ok=True)

if len(args.targets_file) > 0:
  natives_list = open(args.targets_file, 'r').read().split("\n")[:-1]
else:
  natives_list = args.target_list
print("natives_list:", natives_list)

finished_decoys = []
for n in natives_list:
  if os.path.exists(args.output_dir  + args.name + "/results/results_{}.csv".format(n)):
    finished_decoys += [x.split(",")[0] + "_" + x.split(",")[1] for x in open(args.output_dir  + args.name + "/results/results_{}.csv".format(n), "r").readlines()]
finished_decoys = set(finished_decoys)


if os.path.exists(args.output_dir  + args.name + "/finished_targets.txt"):
  finished_targets = set(open(args.output_dir + args.name + "/finished_targets.txt", 'r').read().split("\n")[:-1])
else:
  finished_targets = []

finished_targets = []

unfinished_targets = [n for n in natives_list if n not in finished_targets]
print("unfinished_targets:", unfinished_targets)

# info of the form "target decoy_id"
decoy_list = [x.split() for x in open(os.path.join(args.decoy_dir + "decoy_list.txt"), 'r').readlines()] 
# parse all of the information about the decoys
decoy_data = {}
for field in decoy_fields_list[2:]:
  if os.path.exists(args.decoy_dir + field + ".txt"):
    lines = [x.split() for x in open(args.decoy_dir + field + ".txt", 'r').read().split("\n")[:-1]] # form "target decoy_id metric value"

    # make sure everything is in the same order
    for i,l in enumerate(lines):
      assert l[0] == decoy_list[i][0]
      assert l[1] == decoy_list[i][1]

    decoy_data[field] = [l[-1] for l in lines]
  else:
    decoy_data[field] = [-1]*len(decoy_list) # -1 as a placeholder

decoy_dict = {n : [] for n in natives_list if n not in finished_targets} # key = target name, value = list of Decoy objects

print("decoy_list :", decoy_list)
for i, d in enumerate(decoy_list):

  decoy = Decoy(target=d[0], decoy_id=d[1], decoy_path=args.decoy_dir + "decoy_pdbs/" + d[0] + "/" + d[1], 
                  rmsd = decoy_data["rmsd"][i], rosettascore = decoy_data["rosettascore"][i], gdt_ts = decoy_data["gdt_ts"][i], 
                    tmscore=decoy_data["tmscore"][i], danscore = decoy_data["danscore"][i])

  if decoy.target in decoy_dict and decoy.target + "_" + decoy.decoy_id not in finished_decoys:
    decoy_dict[decoy.target].append(decoy)
print("decoy_dict after decoy_list.txt:", decoy_dict)
# add another decoy entry for the native structure
if args.use_native:
  for n in decoy_dict.keys():
    if n + "_native" not in finished_decoys:
      decoy_dict[n].insert(0, Decoy(target=n, decoy_id="native", decoy_path=args.decoy_dir + "native_pdbs/" + n + "_ignorechain.cif", 
                            rmsd = 0, rosettascore = -1, gdt_ts = 1, tmscore = 1, danscore = -1))

if args.verbose:
  print(finished_decoys)

model_name = "model_{}_ptm".format(args.model_num)
results_key = model_name + "_seed_{}".format(args.seed)
for n in unfinished_targets:

  pdb_native = args.decoy_dir + "native_pdbs/" + n + ".pdb"
  prot_native = protein.from_pdb_string(pdb_to_string(pdb_native))
  seq_native = "".join([residue_constants.restypes[x] for x in prot_native.aatype])
  runner = make_model_runner(model_name, args.recycles)

  # if n + "_none.pdb" not in finished_decoys:

  #   # run the model with no templates
  #   features = make_processed_feature_dict(runner, seq_native, name=n + "_none", seed=args.seed)
  #   result = parse_results(runner.predict(features, random_seed=args.seed), features)

  #   dummy_decoy = Decoy(target=n, decoy_id="none.pdb", decoy_path="_", rmsd=-1, rosettascore=-1, gdt_ts=-1, tmscore=-1,danscore=-1)
  #   write_results(dummy_decoy, result, prot_native=prot_native if args.use_native else None, pdb_native=pdb_native if args.use_native else None)


  # run the model with all of the decoys passed as templates
  print("decoy_dict:", decoy_dict, flush=True)
  
  if args.no_template:
    template = {"template_aatype":np.zeros((1, 1, len(seq_native), 22)),
              "template_all_atom_masks":np.zeros((1, len(seq_native), 37)),
              "template_all_atom_positions":np.zeros((1, len(seq_native), 37, 3)),
              "template_domain_names":np.asarray(["None"])}
    n_repeats = 25
    for i in range(n_repeats):
      if args.use_precomputed_msas:
        features = make_processed_feature_dict(runner, seq_native, name=n + "_no_template", templates=template, seed=i, use_precomputed_msas=True, msa_output_dir=args.msa_dir)
      else:
        features = make_processed_feature_dict(runner, seq_native, name=n + "_no_template", templates=template, seed=i)
        
      prediction_result = runner.predict(features, random_seed=i)
      result = parse_results(prediction_result, features)
      pdb_lines = protein.to_pdb(result["unrelaxed_protein"])
      pdb_out_path = args.output_dir + args.name + "/pdbs/" + n + "_no_template" + str(i) +'.pdb'

      with open(pdb_out_path, 'w') as f:
        f.write(pdb_lines)

  elif args.multiple_decoy:
    n_repeats = 25
    decoy_paths = [d.decoy_path for d in decoy_dict[n]] # all decoy files should be end with .cif
    for i in range(n_repeats):
      result, _ = score_homo_template(seq_native, runner, n + "_multiple", decoy_paths, seed=i)
      pdb_lines = protein.to_pdb(result["unrelaxed_protein"])
      pdb_out_path = args.output_dir + args.name + "/pdbs/" + n + "_multiple_" + str(i) +'.pdb'

      with open(pdb_out_path, 'w') as f:
        f.write(pdb_lines)
  else:
    for d in decoy_dict[n]:
        if d.decoy_path.endswith("cif"):
          result, mismatch = score_homo_template(seq_native, runner, d.target + "_" + d.decoy_id, [d.decoy_path])
        else:
          prot = protein.from_pdb_string(pdb_to_string(d.decoy_path))
          result, mismatch = score_decoy(seq_native, prot, runner, d.target + "_" + d.decoy_id)
        write_results(d, result, prot_native=prot_native if args.use_native else None, pdb_native=pdb_native if args.use_native else None, mismatch=mismatch)


  with open(args.output_dir + args.name + "/finished_targets.txt", 'a') as f:
    f.write(n + "\n")
